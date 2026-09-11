"""验证 run_pipeline 的单实例互斥锁：有活跃实例时应拒绝启动，陈旧锁应自动清理。"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_pipeline import LOCK_PATH, _pid_alive, acquire_lock


def main():
    print(f"锁文件: {LOCK_PATH}")
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)

    # 1) 陈旧锁（用一个几乎不可能存在的 PID）应被清理并成功获取
    io.open(LOCK_PATH, "w", encoding="utf-8").write("999999")
    print("步骤1：写入陈旧锁 pid=999999")
    assert not _pid_alive(999999), "999999 居然存活，换一个 PID"
    acquire_lock()  # 应打印「清理陈旧锁」并接管
    assert io.open(LOCK_PATH, encoding="utf-8").read().strip() == str(os.getpid())
    print("  -> 陈旧锁被清理并接管 [OK]")

    # 2) 模拟「已有活跃实例」：写入当前进程 PID 之外的一个存活 PID
    alive = os.getpid()  # 当前进程自己一定存活
    io.open(LOCK_PATH, "w", encoding="utf-8").write(str(alive + 1) if False else "1")
    # Windows 上 PID 1 通常是 System Idle，改用当前 shell 的父进程更可靠：
    io.open(LOCK_PATH, "w", encoding="utf-8").write(str(os.getppid()))
    print(f"步骤2：写入活跃锁 pid={os.getppid()}（父进程，存活={_pid_alive(os.getppid())}）")
    try:
        acquire_lock()
        raise AssertionError("活跃锁未生效：应当 SystemExit")
    except SystemExit as e:
        msg = str(e)
        assert "已有流水线在运行" in msg, msg
        print("  -> 正确拒绝启动 [OK]")

    # 3) --force-lock 应绕过
    print("步骤3：force=True 应绕过检查")
    acquire_lock(force=True)
    print("  -> 绕过成功 [OK]")

    if os.path.exists(LOCK_PATH):
        os.remove(LOCK_PATH)
    print("\n单实例互斥锁验证通过 [OK]")


if __name__ == "__main__":
    main()
