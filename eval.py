"""
AMR: Asymmetric Memory-decoupled Reconstruction
Standalone Evaluation Script
================================================
Usage:
    # Evaluate on IRDST
    python eval.py --weights weights/amr_irdst_best.pth --config configs/irdst.yaml

    # Evaluate on ITSDT-15K
    python eval.py --weights weights/amr_itsdt_best.pth --config configs/itsdt.yaml

    # With custom thresholds
    python eval.py --weights weights/amr_irdst_best.pth --config configs/irdst.yaml \
        --conf-thresh 0.01 --device cuda:0
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from utils.eval_utils import evaluate_detection, print_evaluation_report
from model.network import MemISTDSmallTarget, build_model
from dataloader import IRDSTDataset, ITSDTDataset, NUDTMIRSDTDataset, irdst_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)




def load_checkpoint(model, weights_path, device):
    """Load model weights from checkpoint."""
    logger.info(f"Loading weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = None
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None:
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    # Strip "module." prefix if present (from DDP training)
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v
               for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(f"{len(missing)} missing keys: "
                       f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected keys: "
                       f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

    if isinstance(ckpt, dict):
        epoch = ckpt.get("epoch", "N/A")
        best_ap = ckpt.get("mAP", "N/A")
        if isinstance(best_ap, float):
            best_ap = f"{best_ap * 100:.2f}%"
        logger.info(f"Checkpoint info: epoch={epoch}, saved_mAP={best_ap}")

    return model


def build_eval_dataset(cfg, split="test"):
    """Build evaluation dataset."""
    dc = cfg.get("data", {})
    dataset_root = dc.get("dataset_root", "dataset/IRDST/IRDST_real")
    input_shape = dc.get("input_shape", [512, 512])
    eval_pad = dc.get("eval_pad_to_size", dc.get("pad_to_size", input_shape))
    pad_divisor = dc.get("pad_divisor", 4)
    dataset_type = dc.get("dataset_type", "irdst").lower()

    common_kwargs = dict(
        image_size=input_shape[0], type=split, augment=False,
        crop_size=dc.get("crop_size", 512), base_size=dc.get("base_size", 512),
        pad_only=dc.get("pad_only", True), pad_to_size=eval_pad,
        pad_divisor=pad_divisor,
        flip_augmentation=False, vflip_augmentation=False,
        brightness_jitter=0.0, contrast_jitter=0.0,
        gaussian_noise=0.0, scale_jitter=0.0,
    )

    if dataset_type == "itsdt":
        return ITSDTDataset(dataset_root=dataset_root, **common_kwargs)
    elif dataset_type == "nudt":
        return NUDTMIRSDTDataset(dataset_root=dataset_root, **common_kwargs)
    return IRDSTDataset(dataset_root=dataset_root, **common_kwargs)


def run_evaluation(model, dataset, device, conf_thresh, iou_thresh, nms_thresh):
    """Run evaluation and return metrics."""
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        results = evaluate_detection(
            model=model, test_dataset=dataset,
            conf_thresh=conf_thresh, nms_threshold=nms_thresh,
            iou_threshold=iou_thresh,
            max_detections=300, device=str(device),
            warmup_iterations=3, local_rank=0, disable_pbar=False,
        )
    return results


def parse_args():
    p = argparse.ArgumentParser(description="AMR Standalone Evaluation")
    p.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--config", default="configs/irdst.yaml", help="Path to config YAML")
    p.add_argument("--split", default="test", choices=["test", "train"])
    p.add_argument("--device", default="cuda:0", help="Device: cuda, cuda:0, cpu")
    p.add_argument("--conf-thresh", type=float, default=None)
    p.add_argument("--iou-thresh", type=float, default=None)
    p.add_argument("--nms-thresh", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    eval_cfg = cfg.get("eval", {})

    conf_thresh = args.conf_thresh if args.conf_thresh is not None \
        else eval_cfg.get("conf_thresh", 0.5)
    iou_thresh = args.iou_thresh if args.iou_thresh is not None \
        else eval_cfg.get("iou_thresh", 0.5)
    nms_thresh = args.nms_thresh if args.nms_thresh is not None \
        else eval_cfg.get("nms_thresh", 0.2)
    logger.info(f"Thresholds: conf={conf_thresh}, iou={iou_thresh}, nms={nms_thresh}")

    # Build model
    model = build_model(cfg)
    model = load_checkpoint(model, args.weights, device)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.2f}M")

    # Build dataset
    dataset = build_eval_dataset(cfg, split=args.split)
    logger.info(f"Eval dataset ({args.split}): {len(dataset)} images")

    # Run evaluation
    t0 = time.perf_counter()
    results = run_evaluation(model, dataset, device, conf_thresh, iou_thresh, nms_thresh)
    wall_time = time.perf_counter() - t0
    logger.info(f"Evaluation finished in {wall_time:.1f}s")

    print()
    print_evaluation_report(results)
    print("\n[SUMMARY]")
    print(f"  Recall    : {results['recall'] * 100:.2f}%")
    print(f"  Precision : {results['precision'] * 100:.2f}%")
    print(f"  F1        : {results['f1'] * 100:.2f}%")
    print(f"  AP@0.5    : {results['ap50'] * 100:.2f}%")
    print(f"  FPS       : {results['fps']:.2f}")
    print(f"  Total GT  : {results['total_gt']}")
    print(f"  TP / FP / FN : {results['tp']} / {results['fp']} / {results['fn']}")


if __name__ == "__main__":
    main()
