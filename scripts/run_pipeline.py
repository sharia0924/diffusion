"""一键实验流水线：DDPM 预训练 → 解码器训练 → 三大护栏评测（P1/P2/P3）+ 鲁棒性全表。

另有 **LDM 隐空间流水线**（`--ldm`，推荐）：VAE → 隐空间 DDPM → 隐空间解码器 → 评测。
像素空间整条链路的上限只有 18.8 dB，隐空间路线可达 35 dB 以上。

用法:
  # 像素空间
  python scripts/run_pipeline.py                          # 全流程（训练+全部评测）
  python scripts/run_pipeline.py --stages ddpm decoder    # 只训练
  python scripts/run_pipeline.py --stages p1 p2 p3        # 只评测（需已有 checkpoint）
  python scripts/run_pipeline.py --stages all --tiny      # 冒烟验证全流程（数分钟）
  python scripts/run_pipeline.py --stages ddpm --ddpm-epochs 300  # 加长训练（自动续训）

  # LDM 隐空间（一条命令跑完）
  python scripts/run_pipeline.py --ldm --stages all
  python scripts/run_pipeline.py --ldm --stages all --finish        # 跑满全部 epoch
  python scripts/run_pipeline.py --ldm --stages vae latent_ddpm     # 只训练前两段
  python scripts/run_pipeline.py --ldm --stages latent_robust latent_p1 latent_p2 latent_p3

阶段（像素空间）:
  ddpm    预训练基础 DDPM            -> checkpoints/ddpm_cifar.pt
  decoder 训练复原解码器             -> checkpoints/decoder.pt (+_best)
  robust  鲁棒性全表                 -> results/robustness.md
  p1      密钥安全表                 -> results/key_security.md
  p2      步数不对称网格+失配曲线    -> results/steps_grid.md, step_mismatch.md
  p3      再生攻击网格+鲁棒性前沿    -> results/regen.md, frontier.md

阶段（LDM，--ldm；产物带 _l32 后缀以免覆盖像素空间结果）:
  vae             训练 VAE（隐空间）        -> checkpoints/vae32.pt
  latent_ddpm     隐空间扩散模型            -> checkpoints/ddpm_latent32.pt
  latent_decoder  隐空间复原解码器(加性注入) -> checkpoints/decoder_latent32_best.pt
  ldm_eval        端到端验收                -> results/ldm_pipeline32.md
  strength_sweep  工作点扫描（质量vs准确率） -> results/strip32.md
  latent_robust   鲁棒性全表                -> results/robustness_l32.md
  latent_p1/p2/p3 三大护栏                  -> results/*_l32.md

特性:
  - **解释器自适应**：若当前解释器没有 torch，自动切换到检测到的可用 conda 环境
    （可用 --python 显式指定），避免 `python` 指向无 torch 的 base 环境时直接失败；
  - 已完成阶段自动跳过（--force 重跑）；ddpm/vae 阶段按 checkpoint 里的 epoch 数
    自动续训（`<out>.last.pt` 断点）；
  - 每阶段日志写入 results/logs/；结束生成 results/pipeline_summary.md；
    任一阶段失败立即终止。
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_ORDER = ["ddpm", "decoder", "robust", "p1", "p2", "p3"]
# LDM（隐空间）阶段：VAE -> 隐空间 DDPM -> 隐空间解码器 -> 评测
LDM_STAGE_ORDER = ["vae", "latent_ddpm", "latent_decoder", "ldm_eval", "strength_sweep",
                   "latent_robust", "latent_p1", "latent_p2", "latent_p3"]

# 常见 conda 环境候选：本机 `python` 可能是没有 torch 的 base
_CONDA_ROOTS = [
    os.path.expandvars(r"%USERPROFILE%\anaconda3\envs"),
    os.path.expandvars(r"%USERPROFILE%\miniconda3\envs"),
    r"D:\anaconda3\envs",
    os.path.expanduser("~/anaconda3/envs"),
    os.path.expanduser("~/miniconda3/envs"),
]
_ENV_NAMES = ["krd-steg", "yolo", "diffusion", "krd"]


def _has_torch(py: str) -> bool:
    try:
        r = subprocess.run([py, "-c", "import torch"], capture_output=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def pick_python(explicit: str | None = None) -> str:
    """选出能 import torch 的解释器：显式指定 > 当前 > 常见 conda 环境。"""
    if explicit:
        if not _has_torch(explicit):
            raise SystemExit(f"[pipeline] --python 指定的解释器无法 import torch: {explicit}")
        return explicit
    if _has_torch(sys.executable):
        return sys.executable
    print(f"[pipeline] 当前解释器无 torch: {sys.executable}\n"
          f"[pipeline] 正在搜索可用的 conda 环境 ...", flush=True)
    py_name = "python.exe" if os.name == "nt" else "bin/python"
    for root in _CONDA_ROOTS:
        if not os.path.isdir(root):
            continue
        # 先试约定名字，再兜底扫全部环境
        names = list(_ENV_NAMES) + [n for n in sorted(os.listdir(root))
                                    if n not in _ENV_NAMES]
        for name in names:
            py = os.path.join(root, name, py_name)
            if os.path.exists(py) and _has_torch(py):
                print(f"[pipeline] 使用解释器: {py}", flush=True)
                return py
    raise SystemExit("[pipeline] 找不到带 torch 的解释器, 请用 --python 指定")


def _force_tolerant_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def probe_mp_ok() -> bool:
    """探测当前环境能否用 multiprocessing 队列（DataLoader num_workers>0 依赖它）。

    受限沙箱/部分 Windows 环境下 `Pipe()` 会抛 PermissionError(WinError 5)，
    此时必须回退到 num_workers=0，否则训练脚本在 DataLoader 构造阶段就直接失败。
    """
    try:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        q.close()
        q.join_thread()
        return True
    except Exception as e:
        print(f"[pipeline] 多进程队列不可用（{type(e).__name__}: {e}）→ 回退 num_workers=0")
        return False


def _pid_alive(pid: int) -> bool:
    """判断进程是否存活（Windows: tasklist；POSIX: kill -0）。"""
    try:
        if os.name == "nt":
            r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                               capture_output=True, text=True, timeout=30)
            return str(pid) in (r.stdout or "")
        os.kill(pid, 0)
        return True
    except Exception:
        return False


LOCK_PATH = os.path.join(BASE, ".pipeline.lock")


def acquire_lock(force: bool = False) -> None:
    """单实例互斥：防止同一时间跑两个流水线。

    两个实例并跑会同时写同一个 checkpoint、抢同一块 GPU 显存
    （实测 4GB 卡上两个 VAE 训练直接把显存打满），必须拦截。

    - force=True（--force-lock）：直接接管锁文件，不做冲突检查；
    - 陈旧锁（PID 已死）自动覆盖；
    - 冲突时打印如何查看/结束已有实例。
    """
    old = 0
    if os.path.exists(LOCK_PATH):
        try:
            old = int(open(LOCK_PATH, encoding="utf-8").read().strip() or 0)
        except Exception:
            old = 0

    if force:
        print(f"[pipeline] --force-lock: 跳过单实例检查（接管锁，原 pid={old or '无'}）")
    elif old and old != os.getpid() and _pid_alive(old):
        raise SystemExit(
            f"[pipeline] 已有流水线在运行（pid={old}）；同时跑两个会争抢 GPU 与\n"
            f"          同一份 checkpoint。请先结束它，或用 --force-lock 强制启动。\n"
            f"          查看进度: Get-Content results\\logs\\*.out -Tail 20\n"
            f"          结束它  : taskkill /PID {old} /T /F")
    elif old:
        print(f"[pipeline] 清理陈旧锁（pid={old} 已不存在）")

    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    import atexit

    def _release():
        try:
            if os.path.exists(LOCK_PATH):
                cur = open(LOCK_PATH, encoding="utf-8").read().strip()
                if cur == str(os.getpid()):
                    os.remove(LOCK_PATH)
        except Exception:
            pass

    atexit.register(_release)


def sh(python: str, script: str, args: list[str], log_path: str) -> None:
    """运行仓库内脚本，输出实时回显并写日志。失败即终止。

    子进程强制 PYTHONUTF8=1：否则 Windows 下子进程按 GBK 写管道，与父进程的
    UTF-8 解码不一致会导致日志乱码。
    """
    cmd = [python, os.path.join("scripts", script)] + args
    print(f"    $ {' '.join(cmd[1:])}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write("    | " + line)
            log.write(line)
            log.flush()
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"\n[pipeline] stage failed (exit {code}), log: {log_path}")


def _ddpm_epochs_done(ckpt: str) -> int:
    """读取 checkpoint 里已完成的 epoch 数（读不到则视为 0）。"""
    if not os.path.exists(ckpt):
        return 0
    try:
        import torch
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        return int(st.get("epochs_done", st.get("epoch", 0)) or 0)
    except Exception:
        return 0


def _need_more(out_ckpt: str, target_epochs: int, suffix: str, key: str) -> bool:
    """训练类阶段的续训判定：断点里的 epoch 数 < 目标 -> 还需要训练。

    用于 vae / latent_ddpm 这类"目标 epoch 会变"的阶段，避免 done_marker
    在目标提高后仍然跳过训练。
    """
    for p in (out_ckpt.replace(".pt", suffix), out_ckpt):
        if os.path.exists(p):
            try:
                import torch
                st = torch.load(p, map_location="cpu", weights_only=False)
                done = int(st.get(key, st.get("epochs_done", 0)) or 0)
                return done < target_epochs
            except Exception:
                continue
    return True


def main():
    _force_tolerant_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["all"],
                    choices=STAGE_ORDER + LDM_STAGE_ORDER + ["all"],
                    help="要执行的阶段（像素空间与 LDM 阶段名都可用）")
    ap.add_argument("--force", action="store_true", help="已完成阶段也强制重跑")
    ap.add_argument("--tiny", action="store_true", help="冒烟模式: 全部参数缩到最小")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")

    # 训练
    ap.add_argument("--ddpm-epochs", type=int, default=60)
    ap.add_argument("--ddpm-batch", type=int, default=128)
    ap.add_argument("--ddpm-lr", type=float, default=2e-4)
    ap.add_argument("--ddpm-base", type=int, default=64)
    ap.add_argument("--ddpm-amp", choices=["auto", "on", "off"], default="off",
                    help="DDPM 混合精度。默认 off：本 U-Net 在 fp16 下会溢出（实测 nan）")
    ap.add_argument("--ddpm-grad-accum", type=int, default=1,
                    help="梯度累积；等效 batch = ddpm-batch × grad-accum")
    ap.add_argument("--ddpm-channels-last", action="store_true",
                    help="channels_last 内存格式（部分显卡卷积更快）")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="开启 cudnn.benchmark（输入尺寸固定时更快）")
    ap.add_argument("--ddpm-save-every", type=int, default=10,
                    help="DDPM 每 N epoch 写一次断点（长时间训练用，支持中断续训）")
    ap.add_argument("--python", default=None,
                    help="显式指定解释器（默认自动挑选带 torch 的环境）")
    ap.add_argument("--decoder-steps", type=int, default=1000)
    ap.add_argument("--decoder-batch", type=int, default=16)
    ap.add_argument("--sched-noise-prob", type=float, default=0.3,
                    help="主解码器的调度坐标加噪概率（P3 的鲁棒性来源）")
    ap.add_argument("--decoder-ablation", action="store_true",
                    help="额外训练 --sched-noise-prob 0 的对照组, 并对 robust/p3 各评一次")

    # 共同评测参数
    ap.add_argument("--hide-steps", type=int, default=50)
    ap.add_argument("--rec-steps", type=int, default=50)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--batch-eval", type=int, default=16)
    ap.add_argument("--n-robust", type=int, default=64)
    ap.add_argument("--n-keysec", type=int, default=32)
    ap.add_argument("--n-grid", type=int, default=24)
    ap.add_argument("--n-regen", type=int, default=32)
    ap.add_argument("--n-frontier", type=int, default=32)
    ap.add_argument("--n-wrong", type=int, default=200)
    ap.add_argument("--regen-t", type=int, default=400)
    ap.add_argument("--nonce-start", type=int, default=0,
                    help="nonce 协议 nonce_i = H(key || nonce-start+i) 的起始计数器；"
                         "所有评测阶段共用，便于复现")
    ap.add_argument("--geom-prob", type=float, default=0.25,
                    help="解码器训练随机失真中几何攻击的条件概率")

    # ---------------- LDM（隐空间）参数 ----------------
    ap.add_argument("--force-lock", action="store_true",
                    help="跳过单实例互斥检查（默认禁止两个流水线并跑）")
    ap.add_argument("--ldm", action="store_true",
                    help="启用 LDM 隐空间流水线（vae -> latent_ddpm -> latent_decoder -> 评测）")
    ap.add_argument("--finish", action="store_true",
                    help="LDM 下跑满全部 epoch（默认用预算内能跑完的 epoch 数）")
    ap.add_argument("--ldm-budget-min", type=float, default=360.0,
                    help="LDM 训练预算（分钟）；据此自动决定 VAE/DDPM 的 epoch 数")
    ap.add_argument("--vae-epochs", type=int, default=25)
    ap.add_argument("--vae-batch", type=int, default=64)
    ap.add_argument("--vae-base", type=int, default=64)
    ap.add_argument("--vae-z-ch", type=int, default=4)
    ap.add_argument("--vae-downsample", type=int, default=1,
                    help="VAE 下采样级数（每级 ×2）；1 -> 32×32 latent（容量达标）")
    ap.add_argument("--vae-ch-mults", default="1,2")
    ap.add_argument("--vae-kl-weight", type=float, default=1e-4)
    ap.add_argument("--vae-resize", type=int, default=64,
                    help="VAE 训练的输入像素尺寸；评测必须用同一尺寸")
    ap.add_argument("--latent-ddpm-epochs", type=int, default=100)
    ap.add_argument("--latent-ddpm-batch", type=int, default=64)
    ap.add_argument("--latent-decoder-steps", type=int, default=1000)
    ap.add_argument("--latent-decoder-batch", type=int, default=16)
    ap.add_argument("--inject-mode", choices=["add", "replace"], default="add",
                    help="注入方式；隐空间必须用 add（replace 实测一加注入 PSNR 即崩到 15dB）")
    ap.add_argument("--latent-strength-min", type=float, default=0.05,
                    help="隐空间解码器训练的 strength 下界（工作点在低强度区）")
    ap.add_argument("--latent-strength-max", type=float, default=0.4,
                    help="隐空间解码器训练的 strength 上界")
    ap.add_argument("--sweep-strengths", default="0.005,0.02,0.05,0.1,0.3,1.0")

    # 各评测专属网格
    ap.add_argument("--grid-hide-list", default="10,25,50,100")
    ap.add_argument("--grid-rec-list", default="10,25,50,100")
    ap.add_argument("--mismatch-rec-list", default="20,30,40,45,50,55,60,70,80")
    ap.add_argument("--regen-t-regs", default="200,400,600,800")
    ap.add_argument("--regen-steps-list", default="25,50")
    ap.add_argument("--frontier-strengths", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = ap.parse_args()

    if args.tiny:  # 冒烟: 全部缩到最小
        args.ddpm_epochs, args.ddpm_batch = min(args.ddpm_epochs, 1), min(args.ddpm_batch, 64)
        args.decoder_steps, args.decoder_batch = min(args.decoder_steps, 30), min(args.decoder_batch, 4)
        args.hide_steps = args.rec_steps = 25
        args.batch_eval = 2
        args.n_robust = args.n_keysec = args.n_grid = args.n_regen = args.n_frontier = 4
        args.n_wrong = 20
        args.grid_hide_list, args.grid_rec_list = "4,10", "4,10"
        args.mismatch_rec_list = "4,10,25"
        args.regen_t_regs, args.regen_steps_list = "400", "25"
        args.frontier_strengths = "0.5,1.0"
        # LDM 也缩到冒烟规模
        args.vae_epochs = min(args.vae_epochs, 2)
        args.vae_base = min(args.vae_base, 32)
        args.vae_resize = 32
        args.vae_downsample = 1
        args.latent_ddpm_epochs, args.latent_ddpm_batch = 2, 64
        args.latent_decoder_steps, args.latent_decoder_batch = 30, 4

    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    log_dir = os.path.join(args.results_dir, "logs")

    python = pick_python(args.python)
    # 单实例互斥：两个流水线并跑会争抢 GPU 与同一份 checkpoint（实测直接打满 4GB 显存）
    acquire_lock(force=args.force_lock)

    order = LDM_STAGE_ORDER if args.ldm else STAGE_ORDER
    stages = order if "all" in args.stages else [s for s in order if s in args.stages]
    if not stages:
        raise SystemExit(f"没有选择任何阶段（可选: {order}）")
    print(f"[pipeline] 模式: {'LDM 隐空间' if args.ldm else '像素空间'}  "
          f"阶段: {' -> '.join(stages)}")
    print(f"[pipeline] ckpt={args.ckpt_dir} results={args.results_dir} "
          f"tiny={args.tiny} python={python}")

    ddpm_ckpt = os.path.join(args.ckpt_dir, "ddpm_cifar.pt")
    summary: list[tuple[str, str, float, str]] = []

    def run_stage(name: str, script: str, script_args: list[str],
                  done_marker: str | None = None) -> str:
        """done_marker 存在且未 --force 时跳过。返回 'done'/'skip'。"""
        if done_marker and os.path.exists(done_marker) and not args.force:
            print(f"[{name}] 已完成, 跳过 ({done_marker})；--force 可重跑")
            return "skip"
        print(f"[{name}] 开始 ...")
        t0 = time.time()
        sh(python, script, script_args, os.path.join(log_dir, f"{name}.log"))
        dt = time.time() - t0
        print(f"[{name}] 完成, 用时 {dt / 60:.1f} min")
        summary.append((name, "done", dt, " ".join(script_args)))
        return "done"

    def decoder_ckpt(tag: str = "main") -> str:
        base = os.path.join(args.ckpt_dir,
                            "decoder.pt" if tag == "main" else f"decoder_{tag}.pt")
        best = base.replace(".pt", "_best.pt")
        return best if os.path.exists(best) else base

    # ---------------- 各阶段 ----------------

    if "ddpm" in stages:
        # 关键：只有当已完成 epoch 数 >= 目标时才跳过。
        # 否则（例如目标从 60 提到 300）继续训练，train_ddpm.py 会自动从 .last.pt 续训。
        done_ep = _ddpm_epochs_done(ddpm_ckpt)
        need_more = done_ep < args.ddpm_epochs
        if need_more:
            print(f"[ddpm] checkpoint 已完成 {done_ep} epoch < 目标 {args.ddpm_epochs}，"
                  f"继续训练（会自动断点续训）")
        run_stage("ddpm", "train_ddpm.py",
                  ["--data-root", args.data_root, "--out", ddpm_ckpt,
                   "--epochs", str(args.ddpm_epochs),
                   "--batch-size", str(args.ddpm_batch),
                   "--lr", str(args.ddpm_lr),
                   "--base", str(args.ddpm_base),
                   "--amp", str(args.ddpm_amp),
                   "--grad-accum", str(args.ddpm_grad_accum),
                   "--save-every", str(args.ddpm_save_every)]
                  + (["--channels-last"] if args.ddpm_channels_last else [])
                  + (["--cudnn-benchmark"] if args.cudnn_benchmark else [])
                  + (["--tiny"] if args.tiny else []),
                  done_marker=None if need_more else ddpm_ckpt)
        if not os.path.exists(ddpm_ckpt):
            raise SystemExit("[pipeline] ddpm 阶段结束但 checkpoint 不存在")

    def train_decoder(tag: str, prob: float):
        out = os.path.join(args.ckpt_dir,
                           "decoder.pt" if tag == "main" else f"decoder_{tag}.pt")
        run_stage(f"decoder[{tag}]", "train_decoder.py",
                  ["--ddpm-ckpt", ddpm_ckpt, "--out", out,
                   "--data-root", args.data_root,
                   "--steps", str(args.decoder_steps),
                   "--batch-size", str(args.decoder_batch),
                   "--sched-noise-prob", str(prob),
                   "--geom-prob", str(args.geom_prob),
                   "--hide-steps", str(args.hide_steps),
                   "--rec-steps", str(args.rec_steps)]
                  + (["--tiny"] if args.tiny else []),
                  done_marker=out)

    if "decoder" in stages:
        train_decoder("main", args.sched_noise_prob)
        if args.decoder_ablation:
            train_decoder("ctrl", 0.0)
        if not os.path.exists(decoder_ckpt("main")):
            raise SystemExit("[pipeline] decoder 阶段被跳过但 checkpoint 不存在")

    # 评测阶段通用参数（nonce 协议统一：nonce_i = H(key || nonce-start+i)）
    common = ["--ddpm-ckpt", ddpm_ckpt, "--data-root", args.data_root,
              "--nonce-start", str(args.nonce_start)]

    def eval_variants(tags: list[str], out_name: str):
        """对每个解码器变体产出带后缀的评测输出。"""
        for tag in tags:
            suffix = "" if tag == "main" else f"_{tag}"
            yield tag, os.path.join(args.results_dir, out_name.replace(".md", f"{suffix}.md"))

    if "robust" in stages:
        tags = ["main", "ctrl"] if args.decoder_ablation else ["main"]
        for tag, out in eval_variants(tags, "robustness.md"):
            run_stage(f"robust[{tag}]", "eval_robustness.py",
                      common + ["--decoder-ckpt", decoder_ckpt(tag), "--out", out,
                                "--n", str(args.n_robust), "--batch", str(args.batch_eval),
                                "--hide-steps", str(args.hide_steps),
                                "--rec-steps", str(args.rec_steps),
                                "--strength", str(args.strength),
                                "--regen-t", str(args.regen_t)],
                      done_marker=out)

    if "p1" in stages:
        out = os.path.join(args.results_dir, "key_security.md")
        run_stage("p1", "eval_key_security.py",
                  common + ["--decoder-ckpt", decoder_ckpt("main"), "--out", out,
                            "--n", str(args.n_keysec), "--batch", str(args.batch_eval),
                            "--n-wrong", str(args.n_wrong),
                            "--hide-steps", str(args.hide_steps),
                            "--rec-steps", str(args.rec_steps),
                            "--strength", str(args.strength)],
                  done_marker=out)

    if "p2" in stages:
        out_grid = os.path.join(args.results_dir, "steps_grid.md")
        run_stage("p2-grid", "eval_steps_grid.py",
                  common + ["--decoder-ckpt", decoder_ckpt("main"), "--out", out_grid,
                            "--n", str(args.n_grid), "--batch", str(args.batch_eval),
                            "--hide-list", args.grid_hide_list,
                            "--rec-list", args.grid_rec_list,
                            "--strength", str(args.strength)],
                  done_marker=out_grid)
        out_mm = os.path.join(args.results_dir, "step_mismatch.md")
        run_stage("p2-mismatch", "eval_step_mismatch.py",
                  common + ["--decoder-ckpt", decoder_ckpt("main"), "--out", out_mm,
                            "--n", str(args.n_grid), "--batch", str(args.batch_eval),
                            "--rec-list", args.mismatch_rec_list,
                            "--hide-steps", str(args.hide_steps),
                            "--strength", str(args.strength)],
                  done_marker=out_mm)

    if "p3" in stages:
        tags = ["main", "ctrl"] if args.decoder_ablation else ["main"]
        for tag, out in eval_variants(tags, "regen.md"):
            run_stage(f"p3-regen[{tag}]", "eval_regen.py",
                      common + ["--decoder-ckpt", decoder_ckpt(tag), "--out", out,
                                "--n", str(args.n_regen), "--batch", str(args.batch_eval),
                                "--rec-steps", str(args.rec_steps),
                                "--t-regs", args.regen_t_regs,
                                "--regen-steps-list", args.regen_steps_list,
                                "--hide-steps", str(args.hide_steps),
                                "--strength", str(args.strength)],
                      done_marker=out)
        out_f = os.path.join(args.results_dir, "frontier.md")
        run_stage("p3-frontier", "plot_frontier.py",
                  common + ["--decoder-ckpt", decoder_ckpt("main"), "--out", out_f,
                            "--n", str(args.n_frontier), "--batch", str(args.batch_eval),
                            "--rec-steps", str(args.rec_steps),
                            "--hide-steps", str(args.hide_steps),
                            "--strengths", args.frontier_strengths],
                  done_marker=out_f)

    # ================= LDM（隐空间）流水线 =================
    if args.ldm:
        vae_ckpt = os.path.join(args.ckpt_dir, "vae32.pt")
        lat_ddpm = os.path.join(args.ckpt_dir, "ddpm_latent32.pt")
        lat_dec = os.path.join(args.ckpt_dir, "decoder_latent32.pt")
        lat_dec_best = lat_dec.replace(".pt", "_best.pt")
        lat_common = ["--ddpm-ckpt", lat_ddpm, "--data-root", args.data_root,
                      "--pixel-res", str(args.vae_resize)]

        def lat_decoder_ckpt() -> str:
            return lat_dec_best if os.path.exists(lat_dec_best) else lat_dec

        def vae_epochs_done(ckpt: str) -> int:
            p = ckpt.replace(".pt", ".last.pt")
            if not os.path.exists(p):
                return 0
            try:
                import torch
                return int(torch.load(p, map_location="cpu",
                                      weights_only=False).get("epoch", 0))
            except Exception:
                return 0

        if "vae" in stages:
            print(f"[vae] 已完成 {vae_epochs_done(vae_ckpt)}/{args.vae_epochs} epoch")
            run_stage("vae", "train_vae.py",
                      ["--data-root", args.data_root, "--out", vae_ckpt,
                       "--epochs", str(args.vae_epochs),
                       "--batch-size", str(args.vae_batch),
                       "--base", str(args.vae_base), "--z-ch", str(args.vae_z_ch),
                       "--downsample", str(args.vae_downsample),
                       "--ch-mults", args.vae_ch_mults,
                       "--kl-weight", str(args.vae_kl_weight),
                       "--resize", str(args.vae_resize),
                       "--save-every", "2", "--log-every", "200"]
                      + (["--tiny"] if args.tiny else []),
                      done_marker=None if _need_more(vae_ckpt, args.vae_epochs, ".last.pt",
                                                     "epoch")
                      else vae_ckpt)

        if "latent_ddpm" in stages:
            if not os.path.exists(vae_ckpt) and not os.path.exists(
                    vae_ckpt.replace(".pt", ".last.pt")):
                raise SystemExit(f"[pipeline] 缺少 VAE checkpoint: {vae_ckpt}")
            vck = vae_ckpt if os.path.exists(vae_ckpt) else vae_ckpt.replace(".pt", ".last.pt")
            run_stage("latent_ddpm", "train_ddpm.py",
                      ["--data-root", args.data_root, "--out", lat_ddpm,
                       "--epochs", str(args.latent_ddpm_epochs),
                       "--batch-size", str(args.latent_ddpm_batch),
                       "--save-every", "10", "--log-every", "200",
                       "--vae-backend", "native", "--vae-ckpt", vck,
                       # **必须与 VAE 训练时的 --resize 一致**：否则 latent 尺寸对不上
                       # （VAE 按 64² 训练 -> latent 32²；若这里漏传，DDPM 会在 16² 上训练，
                       #  评测按 DDPM 的 resize 推断像素尺寸也会错，得到无意义的 PSNR）。
                       "--resize", str(args.vae_resize),
                       "--rebuild-latent-cache"]
                      + (["--tiny"] if args.tiny else []),
                      done_marker=None if _need_more(lat_ddpm, args.latent_ddpm_epochs,
                                                     ".last.pt", "epoch")
                      else lat_ddpm)

        if "latent_decoder" in stages:
            run_stage("latent_decoder", "train_decoder.py",
                      ["--ddpm-ckpt", lat_ddpm, "--out", lat_dec,
                       "--data-root", args.data_root,
                       "--steps", str(args.latent_decoder_steps),
                       "--batch-size", str(args.latent_decoder_batch),
                       "--inject-mode", args.inject_mode,
                       "--strength-min", str(args.latent_strength_min),
                       "--strength-max", str(args.latent_strength_max),
                       "--sched-noise-prob", str(args.sched_noise_prob),
                       "--geom-prob", str(args.geom_prob),
                       "--hide-steps", str(args.hide_steps),
                       "--rec-steps", str(args.rec_steps),
                       "--eval-every", "250"]
                      + (["--tiny"] if args.tiny else []),
                      done_marker=lat_dec)

        if "ldm_eval" in stages:
            out = os.path.join(args.results_dir, "ldm_pipeline32.md")
            run_stage("ldm_eval", "eval_ldm_pipeline.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out,
                                    "--n", str(args.n_grid), "--batch", str(args.batch_eval),
                                    "--hide-steps", str(args.hide_steps),
                                    "--rec-steps", str(args.rec_steps),
                                    "--strengths", "0.02,0.05,0.1,0.3"],
                      done_marker=out)

        if "strength_sweep" in stages:
            out = os.path.join(args.results_dir, "strip32.md")
            run_stage("strength_sweep", "eval_strength_sweep.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out,
                                    "--n", str(args.n_grid), "--batch", str(args.batch_eval),
                                    "--hide-steps", str(args.hide_steps),
                                    "--rec-steps", str(args.rec_steps),
                                    "--strengths", args.sweep_strengths],
                      done_marker=out)

        if "latent_robust" in stages:
            out = os.path.join(args.results_dir, "robustness_l32.md")
            run_stage("latent_robust", "eval_robustness.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out,
                                    "--n", str(args.n_robust),
                                    "--batch", str(args.batch_eval),
                                    "--hide-steps", str(args.hide_steps),
                                    "--rec-steps", str(args.rec_steps),
                                    "--strength", str(args.latent_strength_max)],
                      done_marker=out)

        if "latent_p1" in stages:
            out = os.path.join(args.results_dir, "key_security_l32.md")
            run_stage("latent_p1", "eval_key_security.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out,
                                    "--n", str(args.n_keysec),
                                    "--batch", str(args.batch_eval),
                                    "--n-wrong", str(args.n_wrong),
                                    "--hide-steps", str(args.hide_steps),
                                    "--rec-steps", str(args.rec_steps),
                                    "--strength", str(args.latent_strength_max)],
                      done_marker=out)

        if "latent_p2" in stages:
            out_g = os.path.join(args.results_dir, "steps_grid_l32.md")
            run_stage("latent_p2-grid", "eval_steps_grid.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out_g,
                                    "--n", str(args.n_grid),
                                    "--batch", str(args.batch_eval),
                                    "--hide-list", args.grid_hide_list,
                                    "--rec-list", args.grid_rec_list,
                                    "--strength", str(args.latent_strength_max)],
                      done_marker=out_g)
            out_m = os.path.join(args.results_dir, "step_mismatch_l32.md")
            run_stage("latent_p2-mismatch", "eval_step_mismatch.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out_m,
                                    "--n", str(args.n_grid),
                                    "--batch", str(args.batch_eval),
                                    "--rec-list", args.mismatch_rec_list,
                                    "--hide-steps", str(args.hide_steps),
                                    "--strength", str(args.latent_strength_max)],
                      done_marker=out_m)

        if "latent_p3" in stages:
            out_r = os.path.join(args.results_dir, "regen_l32.md")
            run_stage("latent_p3-regen", "eval_regen.py",
                      lat_common + ["--decoder-ckpt", lat_decoder_ckpt(), "--out", out_r,
                                    "--n", str(args.n_regen),
                                    "--batch", str(args.batch_eval),
                                    "--rec-steps", str(args.rec_steps),
                                    "--t-regs", args.regen_t_regs,
                                    "--regen-steps-list", args.regen_steps_list,
                                    "--hide-steps", str(args.hide_steps),
                                    "--strength", str(args.latent_strength_max)],
                      done_marker=out_r)

    # ---------------- 汇总 ----------------
    summary_path = os.path.join(args.results_dir, "pipeline_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 流水线运行汇总\n\n"
                f"- 模式: **{'LDM 隐空间' if args.ldm else '像素空间'}**\n"
                f"- 阶段: {' -> '.join(stages)}\n- tiny: {args.tiny}\n"
                f"- 解释器: `{python}`\n")
        if args.ldm:
            f.write(f"- VAE: resize={args.vae_resize} downsample={args.vae_downsample} "
                    f"(latent {args.vae_resize // (2 ** args.vae_downsample)}×…) "
                    f"epochs={args.vae_epochs}\n"
                    f"- 隐空间 DDPM: epochs={args.latent_ddpm_epochs}, "
                    f"batch={args.latent_ddpm_batch}\n"
                    f"- 隐空间解码器: steps={args.latent_decoder_steps}, "
                    f"inject_mode=**{args.inject_mode}**, "
                    f"strength∈[{args.latent_strength_min},{args.latent_strength_max}]\n")
        else:
            f.write(f"- DDPM 目标 epoch: {args.ddpm_epochs}（当前 checkpoint 已完成 "
                    f"{_ddpm_epochs_done(ddpm_ckpt)}）\n")
        f.write(f"- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, "
                f"nonce-start={args.nonce_start}\n"
                f"- 几何攻击: 真裁剪/旋转/缩放/平移（crop 不再是 torch.roll 循环平移）\n"
                f"- 再生攻击代价: 同时报告 PSNR vs cover 与 vs stego\n\n"
                "| 阶段 | 状态 | 用时(min) | 命令 |\n|---|---|---|---|\n")
        for name, status, dt, cmd in summary:
            f.write(f"| {name} | {status} | {dt / 60:.1f} | `{cmd}` |\n")
    print(f"\n[pipeline] all stages done [OK]  summary -> {summary_path}")
    for name, status, dt, _ in summary:
        print(f"    {name:<18s} {status}  {dt / 60:6.1f} min")


if __name__ == "__main__":
    main()
