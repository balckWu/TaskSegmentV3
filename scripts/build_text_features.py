from __future__ import annotations
import sys
import argparse
from pathlib import Path


# ============================================================
# 让脚本可以直接通过 python scripts/build_text_features.py 运行
# 不再需要 PYTHONPATH=.
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def parse_args():
    parser = argparse.ArgumentParser(description="Build text features for TaskSegmentV3")

    parser.add_argument(
        "--model-path",
        type=str,
        default="google-bert/bert-large-uncased",
        help=(
            "BERT 模型名称或本地路径。"
            "例如 google-bert/bert-large-uncased、bert-large-uncased，"
            "或者 /home/jiang/models/bert-large-uncased"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./text_features",
        help="文本特征输出目录",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Build Text Features")
    print("=" * 80)
    print(f"项目根目录: {ROOT}")
    print(f"BERT 模型: {args.model_path}")
    print(f"输出目录: {Path(args.output_dir).resolve()}")
    print("=" * 80)

    from tasksegment.text.prompt_encoder import build_all

    build_all(
        model_path=args.model_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()