"""把 lightrag_patch/lightrag/ 下的定制补丁覆盖到已安装的 lightrag 包对应路径。

用法:
    python install_patch.py                # 覆盖到当前解释器已安装的 lightrag 包
    python install_patch.py --dest <路径>  # 覆盖到指定目录（用于测试或非默认环境）
"""
import argparse
import shutil
import sys
from pathlib import Path

PATCH_FILES = [
    "prompt.py",
    "operate.py",
    "rerank.py",
    "llm/openai.py",
    "prompts/__init__.py",
]


def main():
    ap = argparse.ArgumentParser(description="覆盖 LightRAG 定制补丁")
    ap.add_argument("--dest", default=None, help="目标 lightrag 包目录（默认自动探测当前解释器的安装位置）")
    args = ap.parse_args()

    src = Path(__file__).resolve().parent / "lightrag_patch" / "lightrag"
    if not src.exists():
        sys.exit(f"[中止] 补丁目录不存在: {src}")

    if args.dest:
        dest = Path(args.dest).resolve()
    else:
        try:
            import lightrag
        except ImportError:
            sys.exit("[中止] 未找到 lightrag 包——请先 pip install -r requirements.txt")
        dest = Path(lightrag.__file__).resolve().parent

    print(f"补丁来源: {src}")
    print(f"目标目录: {dest}\n")
    for rel in PATCH_FILES:
        f = src / rel
        if not f.exists():
            sys.exit(f"[中止] 补丁文件缺失: {f}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        print(f"  覆盖 {rel}")

    print(f"\n[完成] {len(PATCH_FILES)} 个补丁文件已覆盖。"
          f"缺失其中 prompt.py / operate.py 会导致图谱打分退化，"
          f"缺失 rerank.py 会导致主程序无法启动。")


if __name__ == "__main__":
    main()
