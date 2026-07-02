#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from spaceera.spacealign.trl_rewards import (
    spacealign_absolute_reward,
    spacealign_format_reward,
    spacealign_relative_reward,
    spacealign_task_reward,
)
from spaceera.workflows.data import load_scanforgeqa, write_scanforgeqa
from spaceera.workflows.preparation import prepare_scanforgeqa_samples
from spaceera.workflows.prompts import build_training_prompt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-VL on ScanForgeQA with ScenePick and SpaceAlign GRPO."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--prepared-output", type=Path)
    parser.add_argument("--scene-cache-dir", type=Path, default=Path("outputs/scenepick_cache"))
    parser.add_argument("--vggt-model", help="Required when selected_frames are absent.")
    parser.add_argument("--vggt-device", choices=["cpu", "cuda"])
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spacealign-qwen2.5-vl-7b"))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=16384)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Module names to adapt; change these when using another VLM architecture.",
    )
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Validate data without loading Qwen.")
    return parser


def _prepare_if_needed(args: argparse.Namespace):
    samples = load_scanforgeqa(args.dataset, video_root=args.video_root)
    if any(not sample.selected_frames for sample in samples):
        if not args.vggt_model:
            raise ValueError(
                "dataset contains samples without selected_frames; pass --vggt-model or run "
                "scripts/prepare_scanforgeqa.py first"
            )
        samples = prepare_scanforgeqa_samples(
            samples,
            vggt_model=args.vggt_model,
            cache_dir=args.scene_cache_dir,
            num_frames=args.num_frames,
            max_candidates=args.max_candidates,
            device=args.vggt_device,
        )
        output = args.prepared_output or args.output_dir / "prepared_scanforgeqa.jsonl"
        write_scanforgeqa(output, samples)
    return samples


def _dataset_rows(samples) -> list[Dict[str, Any]]:
    rows = []
    for sample in samples:
        if not sample.selected_frames:
            raise ValueError(f"{sample.sample_id}: ScenePick produced no selected frames")
        rows.append(
            {
                "prompt": build_training_prompt(sample),
                "images": list(sample.selected_frames),
                "solution": sample.answer,
                "question_type": sample.question_type,
                "ground_truth_coordinates_json": json.dumps(
                    sample.ground_truth_coordinates, ensure_ascii=False
                ),
                "category_counts_json": json.dumps(sample.category_counts, ensure_ascii=False),
                "grid_size": 10,
                "alpha": 0.5,
                "sample_id": sample.sample_id,
            }
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    try:
        from datasets import Dataset
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("install training dependencies with: pip install -e '.[training]'") from exc

    samples = _prepare_if_needed(args)
    dataset = Dataset.from_list(_dataset_rows(samples))

    def load_images(batch):
        loaded = []
        for paths in batch["images"]:
            loaded.append([Image.open(path).convert("RGB") for path in paths])
        batch["images"] = loaded
        return batch

    dataset = dataset.with_transform(load_images)
    if args.dry_run:
        first = dataset[0]
        print(
            json.dumps(
                {
                    "samples": len(dataset),
                    "first_sample_id": first["sample_id"],
                    "first_sample_images": len(first["images"]),
                    "has_spatial_supervision": bool(
                        json.loads(first["ground_truth_coordinates_json"])
                    ),
                },
                indent=2,
            )
        )
        return

    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoProcessor
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install training dependencies with: pip install -e '.[training]'") from exc

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    processor.tokenizer.padding_side = "left"
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        gradient_checkpointing=True,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        use_vllm=args.use_vllm,
        model_init_kwargs={
            "torch_dtype": dtype,
            "attn_implementation": args.attn_implementation,
        },
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=[
            spacealign_format_reward,
            spacealign_task_reward,
            spacealign_absolute_reward,
            spacealign_relative_reward,
        ],
        train_dataset=dataset,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
