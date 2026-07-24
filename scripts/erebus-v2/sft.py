#!/usr/bin/env python3
"""
Erebus v2 1.5B — Supervised Fine-Tuning (SFT)

Supports two dataset modes:
  --dataset smoltalk   → general instruct/chat (HuggingFaceTB/smoltalk)
  --dataset xlam       → tool/function calling (Salesforce/xlam-function-calling-60k)

Full fine-tune on the pretrained base model. Loss is masked to assistant tokens only.

Usage (single node, 4 GPUs):
    torchrun --nproc_per_node=4 scripts/erebus-v2/sft.py \
        --model_path /data/checkpoints/final \
        --dataset smoltalk \
        --hf_repo soyrsoyr/erebus-v2-1.5b-instruct
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer


TRAIN_DEFAULTS = dict(
    model_path="soyrsoyr/erebus-v2-1.5b-base",
    dataset="smoltalk",
    seq_len=2048,
    per_device_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    min_lr_ratio=0.0,
    warmup_ratio=0.05,
    weight_decay=0.01,
    max_grad_norm=1.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-8,
    num_epochs=1,
    log_interval=10,
    save_interval=500,
    seed=42,
    bf16=True,
    output_dir="./checkpoints",
    tensorboard_dir="./logs",
    hf_repo=None,
    max_samples=0,
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE_INDEX = -100


def format_smoltalk(example, tokenizer, seq_len):
    """Format a SmolTalk example into token IDs with assistant-only labels."""
    messages = example["messages"]
    if not messages:
        return None

    input_ids = []
    labels = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if not content:
            continue

        header = f"{IM_START}{role}\n"
        footer = f"{IM_END}\n"

        header_ids = tokenizer.encode(header, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = tokenizer.encode(footer, add_special_tokens=False)

        turn_ids = header_ids + content_ids + footer_ids

        if role == "assistant":
            turn_labels = (
                [IGNORE_INDEX] * len(header_ids)
                + content_ids
                + footer_ids
            )
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    if len(input_ids) > seq_len:
        input_ids = input_ids[:seq_len]
        labels = labels[:seq_len]

    if all(l == IGNORE_INDEX for l in labels):
        return None

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def format_xlam(example, tokenizer, seq_len):
    """Format an xLAM function-calling example into token IDs with assistant-only labels."""
    tools_str = example.get("tools", "[]")
    query = example.get("query", "")
    answers_str = example.get("answers", "[]")

    try:
        tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
    except (json.JSONDecodeError, TypeError):
        return None

    if not query or not answers:
        return None

    system_content = (
        "You are a helpful assistant with access to the following tools. "
        "When the user's request requires a tool, respond with a JSON function call.\n\n"
        "Available tools:\n"
    )
    for tool in tools:
        system_content += json.dumps(tool) + "\n"

    tool_calls = []
    for ans in answers:
        call = {"name": ans.get("name", ""), "arguments": ans.get("arguments", {})}
        tool_calls.append(json.dumps(call))
    assistant_content = "\n".join(tool_calls)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_content},
    ]

    return format_smoltalk({"messages": messages}, tokenizer, seq_len)


class SFTDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(ex["input_ids"].size(0) for ex in batch)

    input_ids = []
    labels = []
    attention_mask = []

    for ex in batch:
        seq_len = ex["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([ex["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([ex["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def get_lr(step, total_steps, warmup_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def load_and_format_dataset(args, tokenizer):
    formatter = format_smoltalk if args.dataset == "smoltalk" else format_xlam

    if args.dataset == "smoltalk":
        raw = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    elif args.dataset == "xlam":
        raw = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_samples > 0:
        raw = raw.select(range(min(args.max_samples, len(raw))))

    examples = []
    skipped = 0
    for i, ex in enumerate(raw):
        result = formatter(ex, tokenizer, args.seq_len)
        if result is not None:
            examples.append(result)
        else:
            skipped += 1

    return examples, skipped


def main():
    parser = argparse.ArgumentParser(description="Erebus v2 1.5B SFT")
    for key, default in TRAIN_DEFAULTS.items():
        arg_type = type(default) if default is not None else str
        if arg_type == bool:
            parser.add_argument(f"--{key}", action="store_true", default=default)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=default)
    args = parser.parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16" if args.bf16 else "no",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=args.tensorboard_dir,
    )
    set_seed(args.seed)

    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        print(f"{'='*60}")
        print(f"Erebus v2 1.5B — SFT ({args.dataset})")
        print(f"{'='*60}")
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.dataset}")
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Per-device batch: {args.per_device_batch_size}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        eff_batch = args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        print(f"Effective batch: {eff_batch}")
        print(f"Epochs: {args.num_epochs}")
        print(f"LR: {args.learning_rate}")
        print(f"{'='*60}")

    if is_main:
        print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        print(f"Loading and formatting {args.dataset} dataset...", flush=True)
    examples, skipped = load_and_format_dataset(args, tokenizer)
    if is_main:
        print(f"Formatted {len(examples):,} examples ({skipped:,} skipped)", flush=True)
        avg_len = sum(ex["input_ids"].size(0) for ex in examples) / len(examples)
        print(f"Average sequence length: {avg_len:.0f} tokens", flush=True)

    dataset = SFTDataset(examples)

    steps_per_epoch = math.ceil(len(dataset) / (
        args.per_device_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    ))
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr = args.learning_rate * args.min_lr_ratio

    if is_main:
        print(f"Steps per epoch: {steps_per_epoch:,}")
        print(f"Total steps: {total_steps:,}")
        print(f"Warmup steps: {warmup_steps:,}")
        print(f"{'='*60}")

    if is_main:
        print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Parameters: {param_count:,} ({param_count/1e6:.1f}M)", flush=True)

    pad_id = tokenizer.pad_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
        drop_last=True,
    )

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        fused=True,
    )

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    if is_main:
        os.makedirs(args.tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_dir)

    if is_main:
        print(f"\nStarting SFT training...", flush=True)

    global_step = 0
    start_time = time.time()
    log_start_time = time.time()
    running_loss = 0.0
    log_tokens = 0

    for epoch in range(args.num_epochs):
        if is_main:
            print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---", flush=True)

        for batch_idx, batch in enumerate(dataloader):
            lr = get_lr(global_step, total_steps, warmup_steps, args.learning_rate, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            model.train()

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                running_loss += loss.detach().float().item()
                log_tokens += (batch["attention_mask"].sum()).item()

            if accelerator.sync_gradients:
                grad_norm = 0.0
                if args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if hasattr(grad_norm, "item"):
                        grad_norm = grad_norm.item()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % args.log_interval == 0 and is_main:
                    avg_loss = running_loss / (args.log_interval * args.gradient_accumulation_steps)
                    elapsed = time.time() - log_start_time
                    tps = log_tokens * accelerator.num_processes / elapsed if elapsed > 0 else 0
                    eta = (time.time() - start_time) / global_step * (total_steps - global_step)

                    print(
                        f"step {global_step:>6d}/{total_steps} | "
                        f"epoch {epoch + 1} | "
                        f"loss {avg_loss:.4f} | "
                        f"lr {lr:.2e} | "
                        f"grad {grad_norm:.4f} | "
                        f"tok/s {tps:,.0f} | "
                        f"ETA {eta/3600:.1f}h",
                        flush=True,
                    )
                    writer.add_scalar("sft/loss", avg_loss, global_step)
                    writer.add_scalar("sft/lr", lr, global_step)
                    writer.add_scalar("sft/grad_norm", grad_norm, global_step)
                    writer.add_scalar("sft/tokens_per_sec", tps, global_step)

                    running_loss = 0.0
                    log_start_time = time.time()
                    log_tokens = 0

                if global_step % args.save_interval == 0:
                    save_dir = Path(args.output_dir) / f"sft-step-{global_step}"
                    if is_main:
                        print(f"Saving checkpoint to {save_dir}", flush=True)
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    if is_main:
                        bf16_state = {k: v.to(torch.bfloat16) for k, v in unwrapped.state_dict().items()}
                        unwrapped.save_pretrained(save_dir, safe_serialization=True, state_dict=bf16_state)
                        del bf16_state
                        tokenizer.save_pretrained(save_dir)
                    accelerator.wait_for_everyone()

    final_dir = Path(args.output_dir) / "sft-final"
    if is_main:
        print(f"\nSFT complete! Saving final model to {final_dir}", flush=True)
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if is_main:
        bf16_state = {k: v.to(torch.bfloat16) for k, v in unwrapped.state_dict().items()}
        unwrapped.save_pretrained(final_dir, safe_serialization=True, state_dict=bf16_state)
        del bf16_state
        tokenizer.save_pretrained(final_dir)

        train_info = {
            "dataset": args.dataset,
            "model_path": args.model_path,
            "training_args": vars(args),
            "total_steps": total_steps,
            "num_examples": len(examples),
            "param_count": param_count,
            "gpu_count": accelerator.num_processes,
        }
        with open(final_dir / "training_info.json", "w") as f:
            json.dump(train_info, f, indent=2, default=str)

        writer.close()

        if args.hf_repo:
            print(f"Pushing to HuggingFace: {args.hf_repo}", flush=True)
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(final_dir),
                repo_id=args.hf_repo,
                repo_type="model",
                commit_message=f"Erebus v2 1.5B SFT ({args.dataset})",
            )
            print("Push complete!", flush=True)

    if is_main:
        total_time = time.time() - start_time
        print(f"\nTotal SFT time: {total_time/3600:.2f} hours")
        print(f"Examples: {len(examples):,}")
        print(f"Steps: {global_step:,}")


if __name__ == "__main__":
    main()
