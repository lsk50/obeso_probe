#!/usr/bin/env python3
"""
02_parse_and_tokenize.py
=============================================================
解析 Obeso 数据格式，将字符级标注转换为 token 级标注。

核心任务:
  1. 加载下载的 JSONL 数据
  2. 用 Qwen2.5-7B-Instruct 的 tokenizer 对回复进行 tokenize
  3. 将 annotation 的字符级 index 转换为 token 级 span
  4. 构建训练所需的正负样例定义（事前判断标签）
  5. 保存解析后的数据

输出:
  data/processed/parsed_qwen7b.jsonl        (token 级标注数据)
  data/processed/train_test_split.json      (训练/测试集划分)

依赖:
  pip install transformers
=============================================================
"""
import os
import json
import random
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def char_idx_to_token_idx(char_idx: int, offsets: list) -> int:
    """
    将字符级索引转换为 token 级索引。
    跳过特殊 token 的 None offset。
    """
    best_match = None
    for token_idx, (cs, ce) in enumerate(offsets):
        if cs is None or ce is None:
            continue
        if cs == 0 and ce == 0:
            continue
        if cs <= char_idx < ce:
            return token_idx
        best_match = token_idx
    return best_match if best_match is not None else len(offsets) - 1


def parse_sample(item: dict, tokenizer) -> dict:
    """
    解析单个样本，将字符级标注转换为 token 级标注。
    
    Args:
        item: 原始 JSONL 数据条目
        tokenizer: HuggingFace tokenizer
    
    Returns:
        包含 token 级标注的字典
    """
    # 提取回复文本
    conversation = item["conversation"]
    if len(conversation) < 2:
        return None
    
    response_text = conversation[1]["content"]
    annotations = item.get("annotations", [])
    
    if not response_text or not annotations:
        return None
    
    # Tokenize
    encoding = tokenizer(
        response_text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=4096,  # 限制最大长度
    )
    input_ids = encoding["input_ids"][0]
    offsets = encoding["offset_mapping"][0].tolist()
    
    # 转换标注
    token_annotations = []
    for ann in annotations:
        char_idx = ann.get("index", 0)
        span_text = ann.get("span", "")
        label = ann.get("label", "")
        verification_note = ann.get("verification_note", "")
        
        # 字符位置 → token 位置
        token_start = char_idx_to_token_idx(char_idx, offsets)
        
        # 找到 span 的 token 范围
        span_end_char = char_idx + len(span_text)
        token_end = char_idx_to_token_idx(span_end_char - 1, offsets) + 1
        token_end = min(token_end, len(input_ids))
        
        token_annotations.append({
            "char_index": char_idx,
            "span_text": span_text,
            "label": label,
            "verification_note": verification_note,
            "token_start": token_start,
            "token_end": token_end,
            "token_ids": input_ids[token_start:token_end].tolist(),
        })
    
    # 构建结果
    result = {
        "subset": item.get("subset", ""),
        "model": item.get("model", ""),
        "response_text": response_text,
        "input_ids": input_ids.tolist(),
        "num_tokens": len(input_ids),
        "annotations": token_annotations,
        "n_entities": len(token_annotations),
        "n_not_supported": sum(1 for a in token_annotations if a["label"] == "Not Supported"),
        "n_supported": sum(1 for a in token_annotations if a["label"] == "Supported"),
    }
    
    return result


def build_prehoc_labels(parsed_item: dict, lookahead_tokens: int = 5) -> list:
    """
    构建"事前判断"标签：对每个越界实体，取其前 N 个 token 作为正例。
    
    Args:
        parsed_item: 解析后的样本
        lookahead_tokens: 取越界实体前几个 token 作为正例
    
    Returns:
        训练样本列表 [(token_idx, label, weight, entity_label), ...]
    """
    training_samples = []
    n_tokens = parsed_item["num_tokens"]
    
    # 收集所有实体 token 位置（用于排除负例）
    entity_token_positions = set()
    for ann in parsed_item["annotations"]:
        for t in range(ann["token_start"], ann["token_end"]):
            entity_token_positions.add(t)
    
    for ann in parsed_item["annotations"]:
        s = ann["token_start"]  # span 起始 token 位置
        
        if ann["label"] == "Not Supported":
            # === 正例：越界实体前 N 个 token ===
            for offset in range(1, lookahead_tokens + 1):
                t = s - offset
                if t >= 0:
                    # 紧邻 token 权重高，稍远 token 权重低
                    weight = 1.0 if offset == 1 else 0.5
                    training_samples.append({
                        "token_idx": t,
                        "label": 1,  # 正例
                        "weight": weight,
                        "source": "pre_hoc_positive",
                        "entity": ann["span_text"],
                        "offset_to_entity": offset,
                    })
        else:
            # === 负例：正确实体前 1 个 token ===
            t = s - 1
            if t >= 0 and t not in entity_token_positions:
                training_samples.append({
                    "token_idx": t,
                    "label": 0,  # 负例
                    "weight": 1.0,
                    "source": "supported_entity_preceding",
                    "entity": ann["span_text"],
                    "offset_to_entity": 1,
                })
    
    # === 负例：远离实体的随机 token ===
    non_entity_positions = [
        t for t in range(n_tokens)
        if t not in entity_token_positions
        and all(abs(t - ep) > 10 for ep in entity_token_positions)
    ]
    
    # 每条回复取 5 个远离实体的 token 作为额外负例
    if non_entity_positions:
        n_sample = min(5, len(non_entity_positions))
        random_positions = random.sample(non_entity_positions, n_sample)
        for t in random_positions:
            training_samples.append({
                "token_idx": t,
                "label": 0,
                "weight": 1.0,
                "source": "distant_background",
                "entity": None,
                "offset_to_entity": None,
            })
    
    return training_samples


def train_test_split(n_samples: int, test_ratio: float = 0.2, seed: int = 42):
    """
    按样本划分训练/测试集（避免 token 级泄露）。
    
    Returns:
        train_indices, test_indices
    """
    random.seed(seed)
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    n_test = int(n_samples * test_ratio)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return sorted(train_indices), sorted(test_indices)


def main():
    parser = argparse.ArgumentParser(description="解析 Obeso 数据，转换为 token 级标注")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入 JSONL 文件路径（默认自动查找 Qwen 子集）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"Tokenizer 对应的模型（默认 {MODEL_NAME}）",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=5,
        help="取越界实体前几个 token 作为正例（默认 5）",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集比例（默认 0.2）",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Obeso 数据解析 & Token 级标注转换")
    print("=" * 60)
    
    # 找到输入文件
    if args.input:
        input_path = Path(args.input)
    else:
        # 自动查找 Qwen 子集
        candidates = list(RAW_DIR.glob("longfact_annotations_qwen*.jsonl"))
        if not candidates:
            print(f"[ERROR] 未找到数据文件，请先运行 01_download_obeso_data.py")
            return
        input_path = candidates[0]
    
    print(f"[INFO] 输入文件: {input_path}")
    print(f"[INFO] Tokenizer: {args.model}")
    print(f"[INFO] Lookahead tokens: {args.lookahead}")
    
    # 加载 tokenizer
    print("[INFO] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    # 读取数据
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"[INFO] 共 {len(items)} 条样本")
    
    # 解析每条样本
    parsed_items = []
    total_entities = 0
    total_not_supported = 0
    parse_errors = 0
    
    for i, item in enumerate(items):
        parsed = parse_sample(item, tokenizer)
        if parsed is not None:
            parsed_items.append(parsed)
            total_entities += parsed["n_entities"]
            total_not_supported += parsed["n_not_supported"]
        else:
            parse_errors += 1
        
        if (i + 1) % 100 == 0:
            print(f"  处理中... {i+1}/{len(items)}")
    
    print(f"\n[INFO] 解析完成:")
    print(f"  成功: {len(parsed_items)} 条")
    print(f"  失败: {parse_errors} 条")
    print(f"  总实体数: {total_entities}")
    print(f"  越界实体数: {total_not_supported}")
    
    # 构建事前判断标签
    print(f"\n[INFO] 构建事前判断标签（lookahead={args.lookahead}）...")
    for item in parsed_items:
        item["prehoc_labels"] = build_prehoc_labels(item, args.lookahead)
    
    # 统计正负例
    total_positive = sum(
        1 for item in parsed_items
        for s in item["prehoc_labels"] if s["label"] == 1
    )
    total_negative = sum(
        1 for item in parsed_items
        for s in item["prehoc_labels"] if s["label"] == 0
    )
    print(f"  正例 token 数: {total_positive}")
    print(f"  负例 token 数: {total_negative}")
    print(f"  正负比: 1:{total_negative/max(total_positive,1):.1f}")
    
    # 训练/测试集划分
    train_indices, test_indices = train_test_split(
        len(parsed_items), args.test_ratio
    )
    
    print(f"\n[INFO] 训练/测试集划分 (按样本):")
    print(f"  训练集: {len(train_indices)} 条样本")
    print(f"  测试集: {len(test_indices)} 条样本")
    
    # 保存解析后的数据
    output_path = PROCESSED_DIR / "parsed_qwen7b.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for item in parsed_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[INFO] 解析数据已保存到: {output_path}")
    
    # 保存划分信息
    split_info = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "n_train": len(train_indices),
        "n_test": len(test_indices),
        "test_ratio": args.test_ratio,
        "seed": 42,
        "lookahead_tokens": args.lookahead,
        "total_positive_tokens": total_positive,
        "total_negative_tokens": total_negative,
    }
    split_path = PROCESSED_DIR / "train_test_split.json"
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 划分信息已保存到: {split_path}")
    
    # 打印详细统计
    print(f"\n{'='*60}")
    print(f"  数据解析完成!")
    print(f"{'='*60}")
    print(f"\n  训练集正例: ~{int(total_positive * len(train_indices) / len(parsed_items))}")
    print(f"  训练集负例: ~{int(total_negative * len(train_indices) / len(parsed_items))}")
    print(f"  测试集正例: ~{int(total_positive * len(test_indices) / len(parsed_items))}")
    print(f"  测试集负例: ~{int(total_negative * len(test_indices) / len(parsed_items))}")
    print(f"\n下一步: 运行 03_extract_hidden_states.py 提取隐藏状态")


if __name__ == "__main__":
    main()
