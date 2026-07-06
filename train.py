"""
AMR: Asymmetric Memory-decoupled Reconstruction
Distributed Training Script
================================================
Usage:
    # 4-GPU distributed training
    torchrun --nproc_per_node=4 train.py --config configs/irdst.yaml

    # Single-GPU training
    python train.py --config configs/irdst.yaml

    # Resume from checkpoint
    torchrun --nproc_per_node=4 train.py --config configs/irdst.yaml --resume logs/irdst/20260529_140423
"""

import os
import sys
import json
import random
import logging
import yaml
import argparse
import time as _time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler
from tqdm import tqdm

from utils.eval_utils import evaluate_detection, print_evaluation_report
from model.network import MemISTDSmallTarget, build_model
from model.yolox_loss_optimized import YOLOLossOptimized
from model.losses import ResidualReconstructionLoss
from dataloader import IRDSTDataset, ITSDTDataset, NUDTMIRSDTDataset, irdst_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset(cfg, split="train"):
    """Build dataset based on config."""
    dc = cfg.get("data", {})
    dataset_root = dc.get("dataset_root", "dataset/IRDST/IRDST_real")
    input_shape = dc.get("input_shape", [512, 512])
    pad_to_size = dc.get("pad_to_size", input_shape)
    pad_divisor = dc.get("pad_divisor", 4)
    dataset_type = dc.get("dataset_type", "irdst").lower()

    common_kwargs = dict(
        image_size=input_shape[0], type=split,
        augment=(split == "train") and dc.get("data_augment", True),
        crop_size=dc.get("crop_size", 512), base_size=dc.get("base_size", 512),
        pad_only=dc.get("pad_only", True), pad_to_size=pad_to_size,
        pad_divisor=pad_divisor,
        flip_augmentation=dc.get("flip_augmentation", True) if split == "train" else False,
        vflip_augmentation=dc.get("vflip_augmentation", True) if split == "train" else False,
        brightness_jitter=dc.get("brightness_jitter", 0.2) if split == "train" else 0.0,
        contrast_jitter=dc.get("contrast_jitter", 0.2) if split == "train" else 0.0,
        gaussian_noise=dc.get("gaussian_noise", 0.01) if split == "train" else 0.0,
        scale_jitter=dc.get("scale_jitter", 0.0) if split == "train" else 0.0,
    )

    if dataset_type == "itsdt":
        return ITSDTDataset(dataset_root=dataset_root, **common_kwargs)
    elif dataset_type == "nudt":
        return NUDTMIRSDTDataset(dataset_root=dataset_root, **common_kwargs)
    return IRDSTDataset(dataset_root=dataset_root, **common_kwargs)




def save_checkpoint(model, optimizer, scaler, epoch, save_path, best_ap, cfg, **extra):
    """Save training checkpoint."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "mAP": best_ap,
        "config": cfg,
        **extra,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, save_path)


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    """Load training checkpoint for resume."""
    logger.info(f"Resuming from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", ckpt.get("state_dict", ckpt)))
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt.get("epoch", 0) + 1
    best_ap = ckpt.get("mAP", 0.0)
    logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}, best mAP={best_ap}")
    return start_epoch, best_ap


def parse_args():
    p = argparse.ArgumentParser(description="AMR Distributed Training")
    p.add_argument("--config", default="configs/irdst.yaml", help="Path to config YAML")
    p.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    seed_everything(cfg.get("seed", 42))

    # ---- Dataset ----
    train_dataset = build_dataset(cfg, split="train")
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        collate_fn=irdst_collate_fn,
        pin_memory=True,
    )

    # ---- Model ----
    model = build_model(cfg).to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[Rank {rank}] Model parameters: {total_params / 1e6:.2f}M")

    # ---- Losses ----
    yolox_loss = YOLOLossOptimized(cfg.get("yolox_loss", {}))
    residual_recon_loss = ResidualReconstructionLoss(cfg.get("residual_recon_loss", {}))

    # ---- Optimizer ----
    train_cfg = cfg["training"]
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                            weight_decay=train_cfg.get("weight_decay", 0.01))
    scaler = GradScaler() if train_cfg.get("fp16", False) else None

    # ---- Checkpoint dir ----
    if is_main:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path(train_cfg.get("save_dir", "./logs/default")) / timestamp
        save_dir.mkdir(parents=True, exist_ok=True)
        yaml.dump(cfg, open(save_dir / "config.yaml", "w", encoding="utf-8"))
    else:
        save_dir = None

    # ---- Resume ----
    start_epoch = 0
    best_ap = 0.0
    if args.resume:
        ckpt_path = Path(args.resume) / "last.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(args.resume)  # direct path
        start_epoch, best_ap = load_checkpoint(
            model.module if world_size > 1 else model,
            optimizer, scaler, str(ckpt_path), device,
        )

    # ---- LR Scheduler ----
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["epochs"] - start_epoch,
    )

    # ---- Eval ----
    eval_cfg = cfg.get("eval", {})
    eval_dataset = None
    if eval_cfg.get("eval_flag", False) and is_main:
        eval_dataset = build_dataset(cfg, split="test")

    # ---- Loss scheduling ----
    ls_cfg = cfg.get("loss_scheduler", {})
    recon_start = ls_cfg.get("residual_recon_start_epoch", 0)

    # ---- Training log ----
    training_log = []

    # ---- Training loop ----
    for epoch in range(start_epoch, train_cfg["epochs"]):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = {}

        # Residual reconstruction weight scheduling
        if epoch < recon_start:
            residual_recon_weight = 0.0
        else:
            rampup = ls_cfg.get("residual_recon_rampup_epochs", 1)
            if rampup > 0 and epoch - recon_start < rampup:
                residual_recon_weight = ls_cfg.get("residual_recon_weight", 1.0) * \
                    (epoch - recon_start + 1) / rampup
            else:
                residual_recon_weight = ls_cfg.get("residual_recon_weight", 1.0)

        accumulation_steps = train_cfg.get("accumulation_steps", 1)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}",
                     disable=not is_main)

        for batch_idx, batch in enumerate(pbar):
            images = batch["images"].to(device)
            target_mask = batch["targets_dict"]["target_mask"].to(device)
            yolo_targets = [t.to(device) for t in batch["targets_dict"]["yolo_targets"]]

            with torch.cuda.amp.autocast() if scaler else torch.inference_mode():
                # Forward (with decode for reconstruction loss)
                outputs = (model.module if world_size > 1 else model)(
                    images, decode_image=True
                )
                outputs["target_mask"] = target_mask

                loss_dict = (model.module if world_size > 1 else model).compute_loss(
                    outputs=outputs,
                    labels=yolo_targets,
                    yolox_loss=yolox_loss,
                    residual_recon_loss=residual_recon_loss,
                    residual_recon_weight=residual_recon_weight,
                )

            loss = loss_dict["total_loss"] / accumulation_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # Logging
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            pbar.set_postfix({
                "total": f"{loss_dict['total_loss'].item():.3f}",
                "yolo": f"{loss_dict.get('yolo_loss', 0):.3f}",
                "recon": f"{loss_dict.get('residual_recon_loss', 0):.4f}",
            })

        scheduler.step()

        # Average losses
        n_batches = len(train_loader)
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}

        # ---- Evaluation ----
        eval_results = None
        if eval_dataset is not None and (epoch + 1) % eval_cfg.get("eval_period", 1) == 0:
            model_to_eval = model.module if world_size > 1 else model
            eval_results = evaluate_detection(
                model=model_to_eval,
                test_dataset=eval_dataset,
                conf_thresh=eval_cfg.get("conf_thresh", 0.001),
                nms_threshold=eval_cfg.get("nms_thresh", 0.2),
                iou_threshold=eval_cfg.get("iou_thresh", 0.5),
                max_detections=300,
                device=str(device),
                warmup_iterations=3,
                local_rank=0,
                disable_pbar=False,
            )

        # ---- Save checkpoint ----
        if is_main:
            ap = eval_results["ap50"] if eval_results else 0.0
            is_best = ap > best_ap
            if is_best:
                best_ap = ap

            base_model = model.module if world_size > 1 else model

            # Save last
            save_checkpoint(base_model, optimizer, scaler, epoch,
                            save_dir / "last.pth", best_ap, cfg)

            # Save epoch checkpoint
            if train_cfg.get("save_every_epoch", False):
                save_checkpoint(base_model, optimizer, scaler, epoch,
                                save_dir / f"epoch_{epoch+1:04d}.pth", best_ap, cfg)

            # Save best
            if is_best:
                save_checkpoint(base_model, optimizer, scaler, epoch,
                                save_dir / "best.pth", best_ap, cfg,
                                eval_results=eval_results)

            # Log
            log_entry = {"epoch": epoch + 1, "losses": avg_losses}
            if eval_results:
                log_entry["eval"] = {
                    "recall": eval_results["recall"],
                    "precision": eval_results["precision"],
                    "f1": eval_results["f1"],
                    "ap50": eval_results["ap50"],
                }
                logger.info(f"[Epoch {epoch+1}] mAP={ap*100:.2f}% "
                            f"(best={best_ap*100:.2f}%) "
                            f"P={eval_results['precision']*100:.1f}% "
                            f"R={eval_results['recall']*100:.1f}%")
            training_log.append(log_entry)

            with open(save_dir / "training_log.json", "w") as f:
                json.dump(training_log, f, indent=2, ensure_ascii=False)

    cleanup_distributed()
    if is_main:
        logger.info(f"Training complete. Best mAP: {best_ap*100:.2f}%")
        logger.info(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
