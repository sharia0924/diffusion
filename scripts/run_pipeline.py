"""一键实验流水线：DDPM 预训练 → 解码器训练 → 三大护栏评测（P1/P2/P3）+ 鲁棒性全表。

用法:
  python scripts/run_pipeline.py                          # 全流程（训练+全部评测）
  python scripts/run_pipeline.py --stages ddpm decoder    # 只训练
  python scripts/run_pipeline.py --stages p1 p2 p3        # 只评测（需已有 checkpoint）
  python scripts/run_pipeline.py --stages all --tiny      # 冒烟验证全流程（数分钟）
  python scripts/run_pipeline.py --stages decoder --force # 强制重训解码器
  python scripts/run_pipeline.py --stages all --decoder-ablation  # 额外训练消融对照并评测

阶段:
  ddpm    预训练基础 DDPM            -> checkpoints/ddpm_cifar.pt
  decoder 训练复原解码器             -> checkpoints/decoder.pt (+_best)
          (--decoder-ablation 额外训 --sched-noise-prob 0 对照 -> decoder_ctrl.pt)
  robust  鲁棒性全表                 -> results/robustness.md
  p1      密钥安全表                 -> results/key_security.md
  p2      步数不对称网格+失配曲线    -> results/steps_grid.md, step_mismatch.md
  p3      再生攻击网格+鲁棒性前沿    -> results/regen.md, frontier.md

特性: 已完成阶段自动跳过（--force 重跑）; 每阶段日志写入 results/logs/;
     结束生成 results/pipeline_summary.md; 任一阶段失败立即终止。
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_ORDER = ["ddpm", "decoder", "robust", "p1", "p2", "p3"]


def _force_tolerant_stdio():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def sh(script: str, args: list[str], log_path: str) -> None:
    """运行仓库内脚本，输出实时回显并写日志。失败即终止。

    子进程强制 PYTHONUTF8=1：否则 Windows 下子进程按 GBK 写管道，与父进程的
    UTF-8 解码不一致会导致日志乱码。
    """
    cmd = [sys.executable, os.path.join("scripts", script)] + args
    print(f"    $ {' '.join(cmd[1:])}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1"}
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


def main():
    _force_tolerant_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["all"],
                    choices=STAGE_ORDER + ["all"], help="要执行的阶段")
    ap.add_argument("--force", action="store_true", help="已完成阶段也强制重跑")
    ap.add_argument("--tiny", action="store_true", help="冒烟模式: 全部参数缩到最小")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--results-dir", default="results")

    # 训练
    ap.add_argument("--ddpm-epochs", type=int, default=60)
    ap.add_argument("--ddpm-batch", type=int, default=128)
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

    # 各评测专属网格
    ap.add_argument("--grid-hide-list", default="10,25,50,100")
    ap.add_argument("--grid-rec-list", default="10,25,50,100")
    ap.add_argument("--mismatch-rec-list", default="20,30,40,45,50,55,60,70,80")
    ap.add_argument("--regen-t-regs", default="200,400,600,800")
    ap.add_argument("--regen-steps-list", default="25,50")
    ap.add_argument("--frontier-strengths", default="0.5,0.75,1.0,1.25,1.5,2.0")
    args = ap.parse_args()

    if args.tiny:  # 冒烟: 全部缩到最小, 数分钟跑完全流程
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

    args.ckpt_dir = os.path.abspath(args.ckpt_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    log_dir = os.path.join(args.results_dir, "logs")

    stages = STAGE_ORDER if "all" in args.stages else [s for s in STAGE_ORDER
                                                       if s in args.stages]
    if not stages:
        raise SystemExit("没有选择任何阶段")
    print(f"[pipeline] 阶段: {' -> '.join(stages)}  "
          f"(ckpt={args.ckpt_dir}, results={args.results_dir}"
          f"{', tiny' if args.tiny else ''})")

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
        sh(script, script_args, os.path.join(log_dir, f"{name}.log"))
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
        run_stage("ddpm", "train_ddpm.py",
                  ["--data-root", args.data_root, "--out", ddpm_ckpt,
                   "--epochs", str(args.ddpm_epochs), "--batch-size", str(args.ddpm_batch)]
                  + (["--tiny"] if args.tiny else []),
                  done_marker=ddpm_ckpt)
        if not os.path.exists(ddpm_ckpt):
            raise SystemExit("[pipeline] ddpm 阶段被跳过但 checkpoint 不存在")

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

    # ---------------- 汇总 ----------------
    summary_path = os.path.join(args.results_dir, "pipeline_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 流水线运行汇总\n\n"
                f"- 阶段: {' -> '.join(stages)}\n- tiny: {args.tiny}\n"
                f"- nonce 协议: `H(key || nonce-start+i)`（自包含, 不依赖 cover）, "
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
