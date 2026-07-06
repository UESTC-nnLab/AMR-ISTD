"""
AMR: Asymmetric Memory-decoupled Reconstruction
Single Image Inference Demo
================================================
Usage:
    # Basic inference on a single image
    python demo.py --weights weights/amr_irdst_best.pth --config configs/irdst.yaml \
        --image path/to/image.png

    # With visualization saved
    python demo.py --weights weights/amr_irdst_best.pth --config configs/irdst.yaml \
        --image path/to/image.png --output result.png

    # Batch inference on a directory
    python demo.py --weights weights/amr_irdst_best.pth --config configs/irdst.yaml \
        --image-dir path/to/images/ --output-dir results/
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from model.network import MemISTDSmallTarget, build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(weights_path, config_path, device):
    """Load AMR model with weights."""
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    model = build_model(cfg)
    model = model.to(device)
    model.eval()

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = {k[len("module."):] if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params / 1e6:.2f}M parameters")
    return model, cfg


def preprocess_image(image_path, target_size=(512, 512)):
    """Preprocess image for AMR inference.

    AMR expects 3-channel RGB input with ImageNet normalization.
    """
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # (W, H)

    # Resize to target size
    img = img.resize(target_size, Image.BICUBIC)

    # Normalize with ImageNet stats
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    # HWC -> CHW -> BCHW
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img, original_size


def draw_boxes(image, boxes, scores, color="red", width=2):
    """Draw detection boxes on image."""
    draw = ImageDraw.Draw(image)
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1, max(y1 - 12, 0)), f"{score:.2f}", fill=color)
    return image


def parse_args():
    p = argparse.ArgumentParser(description="AMR Single Image Inference")
    p.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    p.add_argument("--config", default="configs/irdst.yaml", help="Path to config YAML")
    p.add_argument("--image", default=None, help="Path to single input image")
    p.add_argument("--image-dir", default=None, help="Path to directory of images")
    p.add_argument("--output", default=None, help="Output image path (single image mode)")
    p.add_argument("--output-dir", default="outputs/", help="Output directory (batch mode)")
    p.add_argument("--conf-thresh", type=float, default=0.5)
    p.add_argument("--nms-thresh", type=float, default=0.2)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def infer_single(model, image_path, device, conf_thresh, nms_thresh, output_path=None):
    """Run inference on a single image."""
    img_tensor, pil_img, original_size = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        detections = model.detect(img_tensor, conf_thres=conf_thresh, nms_thres=nms_thresh)

    # detections: [B, N, 6] (x1, y1, x2, y2, score, class_id)
    dets = detections[0]

    if dets is not None and len(dets) > 0:
        boxes = dets[:, :4].cpu().numpy()
        scores = dets[:, 4].cpu().numpy()

        logger.info(f"Found {len(boxes)} detections:")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            logger.info(f"  [{i+1}] box={box.tolist()}, score={score:.4f}")

        if output_path:
            result_img = draw_boxes(pil_img, boxes, scores)
            result_img.save(output_path)
            logger.info(f"Result saved to: {output_path}")
    else:
        logger.info("No detections found.")
        if output_path:
            pil_img.save(output_path)

    return dets


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.weights, args.config, device)

    if args.image:
        infer_single(model, args.image, device,
                     args.conf_thresh, args.nms_thresh, args.output)
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")) + \
                    list(img_dir.glob("*.bmp")) + list(img_dir.glob("*.tif"))
        logger.info(f"Processing {len(img_files)} images...")

        for img_path in img_files:
            logger.info(f"Processing: {img_path.name}")
            out_path = out_dir / f"det_{img_path.stem}.png"
            infer_single(model, str(img_path), device,
                         args.conf_thresh, args.nms_thresh, str(out_path))
    else:
        logger.error("Please specify --image or --image-dir")


if __name__ == "__main__":
    main()
