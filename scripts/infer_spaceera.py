#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from spaceera.workflows.data import load_scanforgeqa
from spaceera.workflows.preparation import prepare_scanforgeqa_samples
from spaceera.workflows.prompts import build_inference_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ScenePick + SpatialMind inference with Qwen2.5-VL."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--sample-id")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--vggt-model", help="Required if the sample has no selected_frames.")
    parser.add_argument("--vggt-device", choices=["cpu", "cuda"])
    parser.add_argument("--scene-cache-dir", type=Path, default=Path("outputs/scenepick_cache"))
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter", help="Optional SpaceAlign LoRA checkpoint.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Build inputs without loading Qwen.")
    return parser


def _select_sample(args: argparse.Namespace):
    samples = load_scanforgeqa(args.dataset, video_root=args.video_root)
    if args.sample_id:
        matches = [sample for sample in samples if sample.sample_id == args.sample_id]
        if not matches:
            raise ValueError(f"sample_id not found: {args.sample_id}")
        sample = matches[0]
    else:
        sample = samples[args.index]
    if not sample.selected_frames:
        if not args.vggt_model:
            raise ValueError("sample has no selected_frames; pass --vggt-model")
        sample = prepare_scanforgeqa_samples(
            [sample],
            vggt_model=args.vggt_model,
            cache_dir=args.scene_cache_dir,
            num_frames=args.num_frames,
            max_candidates=args.max_candidates,
            device=args.vggt_device,
        )[0]
    return sample


def main() -> None:
    args = build_parser().parse_args()
    sample = _select_sample(args)
    messages = build_inference_messages(sample)
    if args.dry_run:
        result = {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "selected_frames": sample.selected_frames,
            "messages": messages,
            "dry_run": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Validated inference inputs -> {args.output}")
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("install inference dependencies with: pip install -e '.[training]'") from exc

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=args.min_pixels, max_pixels=args.max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=args.attn_implementation,
    ).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    processor_messages = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        processor_messages.append({"role": message["role"], "content": content})

    try:
        inputs = processor.apply_chat_template(
            processor_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except (TypeError, ValueError, KeyError):
        # Compatibility path for Qwen processor versions that still rely on
        # qwen-vl-utils for external image loading.
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "this processor needs an architecture-specific media loader; "
                "install qwen-vl-utils for Qwen2.5-VL"
            ) from exc
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    trimmed = generated[:, inputs.input_ids.shape[1] :]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    result = {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "selected_frames": sample.selected_frames,
        "response": response,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(response)


if __name__ == "__main__":
    main()
