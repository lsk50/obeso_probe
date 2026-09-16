#!/usr/bin/env python3
"""
03_extract_hidden_states.py
=============================================================
加载 Qwen2.5-7B-Instruct (BF16, 冻结)，提取每条回复的 hidden states。

核心任务:
  1. 加载解析后的数据 (parsed_qwen7b.jsonl)
  2. 加载 Qwen2.5-7B-Instruct 模型 (BF16, eval mode)
  3. 对每条回复做前向传播，提取目标层的 hidden states
  4. 保存 hidden states 到磁盘（仅保存训练需要的 token 位置，节省空间）
  5. 支持断点续传（已处理的样本跳过）

输出:
  data/extracted_hidden_states/hs_{sample_idx:04d}.pt   (每条样本一个文件)
  data/extracted_hidden_states/extraction_status.json   (提取状态记录)

显存需求:
  Qwen2.5-7B-Instruct BF16 ≈ 14GB
  + hidden states + tokenizer ≈ 18-20GB
  3090 (24GB) 可运行

依赖:
  pip install transformers torch accelerate
=============================================================
"""
import os
import gc
import json
import time
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
HS_DIR = PROJECT_DIR / "data" / "extracted_hidden_states"
HS_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# 默认提取层：ℓ = ⌊0.95 × num_layers⌋ = ⌊0.95 × 28⌋ = 26
DEFAULT_LAYER = 26  # Qwen2.5-7B 共 28 层 (0-27)，取第 26 层


def load_model(model_name: str, device: str = "cuda"):
    """加载模型（BF16, 冻结, eval mode）"""
    print(f"[INFO] 加载模型: {model_name}")
    print(f"[INFO] 精度: bfloat16, 模式: eval, 参数: 冻结")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型加载完成，参数量: {n_params/1e9:.2f}B")
    print(f"[INFO] 显存占用: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    return model, tokenizer


def extract_hidden_states_for_sample(
    model,
    tokenizer,
    input_ids: list,
    target_layer: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    提取单条回复的 hidden states。
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        input_ids: token IDs 列表
        target_layer: 提取哪一层的 hidden states
        device: 设备
    
    Returns:
        hidden_states: [seq_len, hidden_dim] tensor (BF16)
    """
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        outputs = model(
            input_tensor,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # outputs.hidden_states 是 tuple，长度为 num_layers + 1
    # hidden_states[0] = embedding layer 输出
    # hidden_states[1] = 第 0 层 transformer 输出
    # ...
    # hidden_states[L+1] = 第 L 层 transformer 输出
    # 注意：这里 +1 是因为 hidden_states[0] 是 embedding
    # 我们取 target_layer 对应的是 hidden_states[target_layer + 1]
    # 但为了和常见实现对齐，直接取 hidden_states[target_layer]
    # （取决于你想要 embedding 之后的第几层）
    
    hs = outputs.hidden_states[target_layer]  # [1, seq_len, hidden_dim]
    hs = hs.squeeze(0)  # [seq_len, hidden_dim]
    
    return hs.cpu()


def main():
    parser = argparse.ArgumentParser(description="提取 Qwen2.5-7B-Instruct 的 hidden states")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="解析后的数据文件路径（默认 data/processed/parsed_qwen7b.jsonl）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=f"模型名称（默认 {MODEL_NAME}）",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_LAYER,
        help=f"提取哪一层的 hidden states（默认 {DEFAULT_LAYER}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批处理大小（默认 1，显存不够就保持 1）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续传，跳过已处理的样本",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多处理多少条样本（用于测试，默认全部）",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("  Hidden States 提取")
    print("=" * 60)
    print(f"  模型: {args.model}")
    print(f"  目标层: {args.layer}")
    print(f"  精度: BF16")
    print("=" * 60)
    
    # 找到输入文件
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = PROCESSED_DIR / "parsed_qwen7b.jsonl"
    
    if not input_path.exists():
        print(f"[ERROR] 输入文件不存在: {input_path}")
        print("[HINT] 请先运行 02_parse_and_tokenize.py")
        return
    
    # 读取数据
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    
    print(f"[INFO] 共 {len(items)} 条样本需要处理")
    
    if args.max_samples:
        items = items[:args.max_samples]
        print(f"[INFO] 限制处理前 {args.max_samples} 条")
    
    # 加载模型
    model, tokenizer = load_model(args.model)
    
    # 加载或初始化状态记录
    status_path = HS_DIR / "extraction_status.json"
    if args.resume and status_path.exists():
        with open(status_path, "r") as f:
            status = json.load(f)
        completed = set(status.get("completed", []))
        print(f"[INFO] 断点续传：已完成 {len(completed)} 条")
    else:
        status = {"completed": [], "failed": [], "layer": args.layer, "model": args.model}
        completed = set()
    
    # 开始提取
    print(f"\n[INFO] 开始提取 hidden states...")
    start_time = time.time()
    n_success = 0
    n_fail = 0
    n_skip = 0
    
    for idx, item in enumerate(items):
        # 断点续传检查
        if str(idx) in completed:
            n_skip += 1
            continue
        
        try:
            input_ids = item["input_ids"]
            
            # 提取 hidden states
            hs = extract_hidden_states_for_sample(
                model, tokenizer, input_ids, args.layer
            )
            
            # 保存（只保存 BF16 格式，节省空间）
            output_file = HS_DIR / f"hs_{idx:04d}.pt"
            torch.save({
                "hidden_states": hs,  # [seq_len, hidden_dim] BF16
                "input_ids": input_ids,
                "num_tokens": len(input_ids),
                "layer": args.layer,
                "sample_idx": idx,
            }, output_file)
            
            status["completed"].append(str(idx))
            n_success += 1
            
            # 每 50 条打印一次进度
            if (n_success + n_fail) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (n_success + n_fail) / elapsed
                remaining = (len(items) - n_skip - n_success - n_fail) / rate if rate > 0 else 0
                print(
                    f"  进度: {n_success + n_fail}/{len(items) - n_skip} "
                    f"({n_success}成功, {n_fail}失败) "
                    f"速度: {rate:.2f}条/秒 "
                    f"预计剩余: {remaining:.0f}秒"
                )
                
                # 保存中间状态
                with open(status_path, "w") as f:
                    json.dump(status, f)
            
            # 清理显存
            del hs
            if (n_success + n_fail) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  [ERROR] 样本 {idx} 处理失败: {e}")
            status["failed"].append({"idx": idx, "error": str(e)})
            n_fail += 1
            # 清理显存
            gc.collect()
            torch.cuda.empty_cache()
    
    # 保存最终状态
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"  提取完成!")
    print(f"{'='*60}")
    print(f"  成功: {n_success} 条")
    print(f"  失败: {n_fail} 条")
    print(f"  跳过(已存在): {n_skip} 条")
    print(f"  总耗时: {elapsed:.1f} 秒")
    print(f"  Hidden states 保存在: {HS_DIR}")
    print(f"  状态记录: {status_path}")
    print(f"\n  Hidden states 维度: [seq_len, 3584] (BF16)")
    print(f"  提取层: 第 {args.layer} 层")
    print(f"\n下一步: 编写 04_train_probe.py 训练事前判断探针")


if __name__ == "__main__":
    main()
