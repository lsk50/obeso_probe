#!/usr/bin/env python3
"""
01_download_obeso_data.py
=============================================================
从 HuggingFace 下载 Obeso 长文本幻觉检测数据集。

数据集地址:
  https://huggingface.co/datasets/obalcells/longfact-annotations

本脚本下载 Qwen2.5-7B-Instruct 子集（主实验用），
同时提供可选参数下载其他模型子集（用于跨模型对比实验）。

输出:
  data/raw/longfact_annotations_qwen7b.jsonl    (Qwen 子集)
  data/raw/longfact_annotations_{model}.jsonl   (其他子集，可选)
  data/raw/dataset_stats.json                   (数据集统计信息)

依赖:
  pip install datasets huggingface_hub
=============================================================
"""
import os
import json
import argparse
from pathlib import Path

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 所有可用的模型子集
AVAILABLE_SUBSETS = [
    "Llama-3.3-70B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Gemma-2-9B-IT",
    "Mistral-Small-24B",
    "Qwen2.5-7B-Instruct",
]

DATASET_NAME = "obalcells/longfact-annotations"


def download_subset(subset_name: str, output_path: Path):
    """下载指定子集并保存为 JSONL 格式"""
    print(f"\n{'='*60}")
    print(f"  下载子集: {subset_name}")
    print(f"{'='*60}")

    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_NAME, subset_name, split="train")
        print(f"[INFO] 加载成功，共 {len(ds)} 条样本")
    except Exception as e:
        print(f"[ERROR] 加载失败: {e}")
        print("[HINT] 确保已安装: pip install datasets")
        print("[HINT] 如果是网络问题，可设置镜像: export HF_ENDPOINT=https://hf-mirror.com")
        return None

    # 保存为 JSONL
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            record = {
                "subset": item.get("subset", ""),
                "model": item.get("model", subset_name),
                "conversation": item.get("conversation", []),
                "annotations": item.get("annotations", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[INFO] 已保存 {count} 条样本到: {output_path}")
    return count


def compute_stats(all_stats: dict, output_path: Path):
    """计算并保存数据集统计信息"""
    print(f"\n{'='*60}")
    print(f"  数据集统计")
    print(f"{'='*60}")

    summary = {}
    for subset, stats in all_stats.items():
        total_annotations = 0
        supported_count = 0
        not_supported_count = 0
        insufficient_count = 0
        total_response_length = 0

        for item in stats["items"]:
            total_annotations += len(item["annotations"])
            for ann in item["annotations"]:
                label = ann.get("label", "")
                if label == "Supported":
                    supported_count += 1
                elif label == "Not Supported":
                    not_supported_count += 1
                elif label == "Insufficient Information":
                    insufficient_count += 1

            response = item["conversation"][1]["content"] if len(item["conversation"]) > 1 else ""
            total_response_length += len(response)

        n_samples = len(stats["items"])
        avg_annotations = total_annotations / n_samples if n_samples > 0 else 0
        avg_response_length = total_response_length / n_samples if n_samples > 0 else 0

        summary[subset] = {
            "n_samples": n_samples,
            "total_annotations": total_annotations,
            "avg_annotations_per_sample": round(avg_annotations, 2),
            "supported_entities": supported_count,
            "not_supported_entities": not_supported_count,
            "insufficient_info_entities": insufficient_count,
            "avg_response_length_chars": round(avg_response_length, 1),
            "hallucination_rate": round(
                not_supported_count / total_annotations * 100, 2
            ) if total_annotations > 0 else 0,
        }

        print(f"\n  [{subset}]")
        print(f"    样本数: {n_samples}")
        print(f"    总实体标注数: {total_annotations}")
        print(f"    平均每样本实体数: {avg_annotations:.2f}")
        print(f"    Supported: {supported_count}")
        print(f"    Not Supported: {not_supported_count}")
        print(f"    Insufficient Info: {insufficient_count}")
        print(f"    幻觉率: {summary[subset]['hallucination_rate']}%")
        print(f"    平均回复长度(字符): {avg_response_length:.1f}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] 统计信息已保存到: {output_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="下载 Obeso 长文本幻觉标注数据")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Qwen2.5-7B-Instruct"],
        choices=AVAILABLE_SUBSETS,
        help="要下载的模型子集（可多选）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有子集",
    )
    args = parser.parse_args()

    subsets_to_download = AVAILABLE_SUBSETS if args.all else args.subsets

    print("=" * 60)
    print("  Obeso 长文本幻觉标注数据下载")
    print(f"  数据集: {DATASET_NAME}")
    print(f"  子集: {subsets_to_download}")
    print("=" * 60)

    all_stats = {}
    for subset in subsets_to_download:
        # 生成输出文件名
        safe_name = subset.lower().replace(".", "").replace("-", "_")
        output_path = RAW_DIR / f"longfact_annotations_{safe_name}.jsonl"

        # 下载
        count = download_subset(subset, output_path)

        if count is not None:
            # 读取保存的数据用于统计
            items = []
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    items.append(json.loads(line))
            all_stats[subset] = {"items": items}

    # 计算统计信息
    if all_stats:
        compute_stats(all_stats, RAW_DIR / "dataset_stats.json")

    print("\n" + "=" * 60)
    print("  下载完成!")
    print("=" * 60)
    print(f"\n数据文件保存在: {RAW_DIR}")
    print(f"\n下一步: 运行 02_parse_and_tokenize.py 进行数据解析和 token 化")


if __name__ == "__main__":
    main()
