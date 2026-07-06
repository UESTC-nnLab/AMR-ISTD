"""
Evaluation utilities for infrared small target detection.
No OpenCV dependency - pure NumPy / PyTorch / PIL implementation.

AP@0.5 uses VOC 2010+ All-point interpolation (AUC).
Matches MoPKL / DNANet evaluation protocol.
"""

import os
import time
from typing import Optional
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ===========================================================================
# IoU helpers
# ===========================================================================

def compute_iou_single(
    box: np.ndarray,
    gt_boxes: np.ndarray,
) -> np.ndarray:
    """
    Vectorised IoU: one prediction box vs all GT boxes.
    box:      (4,)   x1,y1,x2,y2
    gt_boxes: (N, 4) x1,y1,x2,y2
    returns:  (N,)   float64
    """
    if len(gt_boxes) == 0:
        return np.zeros(0, dtype=np.float64)
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_gt  = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = area_box + area_gt - inter
    return np.where(union > 0.0, inter / union, 0.0).astype(np.float64)


def compute_iou(
    box1: Union[np.ndarray, List],
    box2: Union[np.ndarray, List],
) -> float:
    """IoU between two single boxes (x1,y1,x2,y2). Kept for backward compat."""
    b1 = np.asarray(box1, dtype=np.float64)
    b2 = np.asarray(box2, dtype=np.float64).reshape(1, 4)
    res = compute_iou_single(b1, b2)
    return float(res[0]) if len(res) > 0 else 0.0


# ===========================================================================
# AP  -  VOC 2010+ / COCO  All-point interpolation  (AUC)
# ===========================================================================

def compute_ap(
    scores:   Union[List[float], np.ndarray],
    tp_flags: Union[List[int],   np.ndarray],
    num_gt:   int,
    return_curve: bool = False,
):
    """
    Average Precision via All-point interpolation (VOC 2010+ / COCO / MoPKL).

    Algorithm
    ---------
    1. Sort detections by descending confidence.
    2. Compute cumulative TP/FP -> precision & recall at each rank.
    3. Envelope: make precision monotonically non-increasing (right-to-left).
    4. Sentinel points: prepend (r=0, p=1) and append (r=last_recall, p=0).
    5. AUC = sum of rectangular slabs at every recall-level change.

    Strictly more accurate than the VOC 2007 11-point approximation.
    Matches sklearn.metrics.average_precision_score and MoPKL protocol.

    Returns
    -------
    If return_curve=False: float AP value.
    If return_curve=True: (ap, pr_curve_dict) where pr_curve_dict has keys
      'recall' and 'precision' (numpy arrays ready for plotting).
    """
    if len(scores) == 0 or num_gt == 0:
        if return_curve:
            return 0.0, {"recall": np.array([0.0, 1.0]), "precision": np.array([1.0, 0.0])}
        return 0.0

    scores   = np.asarray(scores,   dtype=np.float64)
    tp_flags = np.asarray(tp_flags, dtype=np.float64)

    order    = np.argsort(-scores)
    tp_flags = tp_flags[order]

    tp_cum = np.cumsum(tp_flags)
    fp_cum = np.cumsum(1.0 - tp_flags)

    recall    = tp_cum / float(num_gt)
    precision = tp_cum / (tp_cum + fp_cum)

    # Sentinels: start at (0,1), end at (last_recall, 0)
    mrec = np.concatenate(([0.0], recall,    [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Monotone non-increasing envelope (right-to-left cummax)
    np.maximum.accumulate(mpre[::-1], out=mpre[::-1])

    # Integration: rectangular slabs at recall-change points
    change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))

    if return_curve:
        return ap, {"recall": mrec, "precision": mpre}
    return ap


# ===========================================================================
# Dataset-level metrics  (single pass — used internally by evaluate_detection)
# ===========================================================================

def compute_metrics(
    all_predictions: List[np.ndarray],
    all_targets:     List[np.ndarray],
    iou_threshold:   float = 0.5,
    conf_thresh:  float = 0.05,
) -> Dict[str, float]:
    """
    Single-pass detection metrics over the entire dataset.

    Design
    ------
    - GT conversion (xywh -> xyxy), confidence filtering, greedy matching,
      TP/FP/FN counting, and AP score accumulation are all done in ONE loop.
    - AP computed with All-point AUC via compute_ap().
    - No duplicate traversal of predictions.

    Parameters
    ----------
    all_predictions : list of (M_i, 5) float arrays  [x1,y1,x2,y2,score]
    all_targets     : list of (N_i, 4) float arrays  [x,y,w,h] xywh format
    iou_threshold   : IoU threshold for TP decision
    conf_thresh  : minimum score to consider a prediction
    """
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    total_gt: int = 0
    all_scores:   List[float] = []
    all_tp_flags: List[int]   = []

    for preds, targets in zip(all_predictions, all_targets):
        # ---- GT: xywh -> xyxy -----------------------------------------------
        if len(targets) > 0:
            gt = np.asarray(targets, dtype=np.float64)
            if gt.ndim == 2 and gt.shape[1] >= 4:
                gt_boxes = np.stack(
                    [gt[:, 0],              gt[:, 1],
                     gt[:, 0] + gt[:, 2],  gt[:, 1] + gt[:, 3]], axis=1)
            else:
                gt_boxes = gt[:, :4].copy()
            num_gt = len(gt_boxes)
        else:
            gt_boxes = np.zeros((0, 4), dtype=np.float64)
            num_gt   = 0
        total_gt += num_gt

        # ---- Filter by confidence -------------------------------------------
        if len(preds) > 0:
            keep = preds[:, 4] >= conf_thresh
            preds = preds[keep]
        if len(preds) == 0:
            total_fn += num_gt          # all GT unmatched
            continue

        pred_boxes  = preds[:, :4].astype(np.float64)
        pred_scores = preds[:, 4].astype(np.float64)

        # ---- Greedy matching (descending score order) -----------------------
        gt_matched = np.zeros(num_gt, dtype=bool)
        order      = np.argsort(-pred_scores)

        for idx in order:
            score = float(pred_scores[idx])
            if num_gt > 0:
                ious            = compute_iou_single(pred_boxes[idx], gt_boxes)
                ious[gt_matched] = -1.0          # mask already-matched GT
                best_gt  = int(np.argmax(ious))
                best_iou = float(ious[best_gt])
            else:
                best_gt  = -1
                best_iou = 0.0

            all_scores.append(score)
            if best_iou >= iou_threshold:
                gt_matched[best_gt] = True
                total_tp += 1
                all_tp_flags.append(1)
            else:
                total_fp += 1
                all_tp_flags.append(0)

        total_fn += int(np.sum(~gt_matched))

    # ---- Dataset-level summary ----------------------------------------------
    recall    = total_tp / max(total_gt, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    f1        = 2.0 * precision * recall / max(precision + recall, 1e-9)
    ap50, pr_curve = compute_ap(all_scores, all_tp_flags, total_gt, return_curve=True)

    return {
        "recall":    recall,
        "precision": precision,
        "f1":        f1,
        "ap50":      ap50,
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
        "total_gt":  total_gt,
        "pr_curve":  pr_curve,
    }


# ===========================================================================
# Main evaluation entry point
# ===========================================================================

def _parse_detections(det_tensor: torch.Tensor, conf_thresh: float) -> np.ndarray:
    """
    Convert model.detect() output for one image to a clean (M, 5) numpy array.

    model.detect() returns [max_detections, 6] padded with zeros.
    Real detections have score > 0; we additionally apply conf_thresh.

    Parameters
    ----------
    det_tensor    : (max_detections, 6) float tensor [x1,y1,x2,y2,score,cls]
    conf_thresh: minimum score to keep

    Returns
    -------
    (M, 5) float64 array [x1,y1,x2,y2,score], M may be 0.
    """
    if det_tensor is None:
        return np.zeros((0, 5), dtype=np.float64)
    arr = det_tensor.cpu().numpy().astype(np.float64)   # (max_det, 6)
    # strip zero-padded rows (score == 0 means padding)
    valid = arr[:, 4] >= max(conf_thresh, 1e-9)
    return arr[valid, :5]                                # keep x1y1x2y2+score


def evaluate_detection(
    model,
    test_dataset,
    conf_thresh:      float = 0.05,
    nms_threshold:       float = 0.5,
    iou_threshold:       float = 0.5,
    max_detections:      int   = 300,
    device:              str   = "cuda",
    warmup_iterations:   int   = 3,
    local_rank:          int   = 0,
    disable_pbar:        bool  = False,
    save_visualizations: bool  = False,
    output_dir:          str   = None,
    show_score:          bool  = False,
    save_detections:     bool  = False,
    detection_dir:       str   = None,
    custom_collate_fn:   Optional[object] = None,
) -> Dict[str, float]:
    """
    Run inference over *test_dataset* and compute detection metrics.

    Design
    ------
    - Inference (model.detect) and result collection run in one forward pass
      per image — no second loop over predictions.
    - Metric computation is delegated entirely to compute_metrics(), which
      performs a single joint pass for TP/FP/FN counting AND AP accumulation
      (no duplication).
    - FPS timing uses CUDA events when on GPU for accuracy.
    - Optional PIL-based visualisation (boxes drawn on grayscale images).

    Parameters
    ----------
    model                : MemISTDSmallTarget (or DDP-wrapped).
    test_dataset         : IRDSTDataset.
    conf_thresh       : minimum objectness score kept after NMS.
    nms_threshold        : NMS IoU threshold (passed to model.detect).
    iou_threshold        : IoU threshold for TP/FP decision.
    max_detections       : cap on detections per image.
    device               : torch device string.
    warmup_iterations    : warm-up forward passes (not timed).
    local_rank           : used for pbar suppression on non-main ranks.
    disable_pbar         : suppress tqdm progress bar.
    save_visualizations  : draw and save detection images.
    output_dir           : directory for visualisation images.
    show_score           : overlay confidence scores on drawn boxes.
    save_detections      : save detection result images with white boxes.
    detection_dir        : directory for detection result images.

    Returns
    -------
    dict with keys: recall, precision, f1, ap50, fps,
                    avg_inference_time_ms, total_images,
                    total_gt, tp, fp, fn.
    """
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    from dataloader import irdst_collate_fn
    collate_fn = custom_collate_fn if custom_collate_fn is not None else irdst_collate_fn
    loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if save_visualizations:
        if output_dir is None:
            output_dir = "./eval_vis"
        os.makedirs(output_dir, exist_ok=True)
    
    if save_detections:
        if detection_dir is None:
            detection_dir = "./results"
        os.makedirs(detection_dir, exist_ok=True)

    # ---- warm-up ------------------------------------------------------------
    warmup_done = 0
    with torch.no_grad():
        for images, _ in loader:
            if warmup_done >= warmup_iterations:
                break
            images = images.to(device)
            raw_model.detect(
                images,
                conf_thres=conf_thresh,
                nms_thres=nms_threshold,
                max_detections=max_detections,
            )
            warmup_done += 1

    # ---- inference + collection ---------------------------------------------
    all_predictions: List[np.ndarray] = []
    all_targets:     List[np.ndarray] = []

    use_cuda_timing = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda_timing:
        starter      = torch.cuda.Event(enable_timing=True)
        ender        = torch.cuda.Event(enable_timing=True)
        total_gpu_ms = 0.0
    else:
        total_wall_s = 0.0

    n_images   = 0
    img_index  = 0

    running_tp = 0
    running_fp = 0
    running_fn = 0
    running_gt = 0

    pbar = tqdm(
        loader,
        desc="Evaluating",
        disable=disable_pbar,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, targets_dict in pbar:
            images = images.to(device, non_blocking=True)
            bs     = images.shape[0]

            if use_cuda_timing:
                starter.record()
                det_batch = raw_model.detect(
                    images,
                    conf_thres=conf_thresh,
                    nms_thres=nms_threshold,
                    max_detections=max_detections,
                )
                ender.record()
                torch.cuda.synchronize()
                total_gpu_ms += starter.elapsed_time(ender)
            else:
                t0 = time.perf_counter()
                det_batch = raw_model.detect(
                    images,
                    conf_thres=conf_thresh,
                    nms_thres=nms_threshold,
                    max_detections=max_detections,
                )
                total_wall_s += time.perf_counter() - t0

            n_images += bs

            gt_boxes_batch = targets_dict.get("boxes", [])
            letterbox_info_batch = targets_dict.get("letterbox_info", None)

            for i in range(bs):
                det_i = _parse_detections(
                    det_batch[i] if (det_batch is not None and i < len(det_batch)) else None,
                    conf_thresh,
                )

                if i < len(gt_boxes_batch):
                    gt_i = gt_boxes_batch[i]
                    if isinstance(gt_i, torch.Tensor):
                        gt_i = gt_i.cpu().numpy()
                    gt_i = np.asarray(gt_i, dtype=np.float64)
                    if gt_i.ndim == 1:
                        gt_i = gt_i.reshape(1, -1) if gt_i.size > 0 else np.zeros((0, 4), dtype=np.float64)
                else:
                    gt_i = np.zeros((0, 4), dtype=np.float64)
                
                letterbox_info = None
                if letterbox_info_batch is not None:
                    if isinstance(letterbox_info_batch, list) and i < len(letterbox_info_batch):
                        letterbox_info = letterbox_info_batch[i]
                    elif isinstance(letterbox_info_batch, dict):
                        letterbox_info = letterbox_info_batch

                all_predictions.append(det_i)
                all_targets.append(gt_i)

                n_gt = len(gt_i)
                n_det = len(det_i)
                running_gt += n_gt

                if n_det > 0 and n_gt > 0:
                    gt_xyxy = np.zeros((n_gt, 4), dtype=np.float64)
                    gt_xyxy[:, 0] = gt_i[:, 0]
                    gt_xyxy[:, 1] = gt_i[:, 1]
                    gt_xyxy[:, 2] = gt_i[:, 0] + gt_i[:, 2]
                    gt_xyxy[:, 3] = gt_i[:, 1] + gt_i[:, 3]

                    det_xyxy = det_i[:, :4]

                    gt_matched = np.zeros(n_gt, dtype=bool)
                    order = np.argsort(-det_i[:, 4])

                    for idx in order:
                        ious = compute_iou_single(det_xyxy[idx], gt_xyxy)
                        ious[gt_matched] = -1.0
                        best_gt = int(np.argmax(ious))
                        best_iou = float(ious[best_gt])

                        if best_iou >= iou_threshold:
                            gt_matched[best_gt] = True
                            running_tp += 1
                        else:
                            running_fp += 1

                    running_fn += int(np.sum(~gt_matched))
                elif n_det > 0:
                    running_fp += n_det
                elif n_gt > 0:
                    running_fn += n_gt

                prec = running_tp / max(running_tp + running_fp, 1) * 100
                rec = running_tp / max(running_gt, 1) * 100
                pbar.set_postfix({
                    'TP': running_tp,
                    'FP': running_fp,
                    'FN': running_fn,
                    'P': f'{prec:.1f}%',
                    'R': f'{rec:.1f}%'
                })

                if save_visualizations and HAS_PIL:
                    _draw_and_save(
                        image_tensor=images[i],
                        det_boxes=det_i,
                        gt_xywh=gt_i,
                        out_path=os.path.join(output_dir, f"{img_index:06d}.png"),
                        show_score=show_score,
                    )

                if save_detections and HAS_PIL:
                    _save_detection_result(
                        image_tensor=images[i],
                        det_boxes=det_i,
                        out_path=os.path.join(detection_dir, f"{img_index:06d}.png"),
                        letterbox_info=letterbox_info,
                    )

                img_index += 1

    # ---- compute metrics (single pass, no duplication) ----------------------
    metrics = compute_metrics(
        all_predictions=all_predictions,
        all_targets=all_targets,
        iou_threshold=iou_threshold,
        conf_thresh=conf_thresh,
    )

    # ---- FPS / timing -------------------------------------------------------
    if n_images > 0:
        if use_cuda_timing:
            total_ms = total_gpu_ms
        else:
            total_ms = total_wall_s * 1000.0
        fps                        = n_images / (total_ms / 1000.0)
        avg_inference_time_ms      = total_ms / n_images
    else:
        fps                        = 0.0
        avg_inference_time_ms      = 0.0

    metrics["fps"]                  = fps
    metrics["avg_inference_time_ms"] = avg_inference_time_ms
    metrics["total_images"]          = n_images
    return metrics


# ===========================================================================
# Visualisation helper  (PIL-based, optional)
# ===========================================================================

def _draw_and_save(
    image_tensor,
    det_boxes:  np.ndarray,
    gt_xywh:    np.ndarray,
    out_path:   str,
    show_score: bool = False,
) -> None:
    """
    Draw GT boxes (green) and predicted boxes (red) on the image and save.

    Parameters
    ----------
    image_tensor : (3, H, W) or (1, H, W) or (H, W) float tensor, ImageNet normalized
    det_boxes    : (M, 5) float array [x1,y1,x2,y2,score]
    gt_xywh      : (N, 4) float array [x,y,w,h]
    out_path     : output PNG path
    show_score   : if True, draw score text next to each detection
    """
    if not HAS_PIL:
        return

    img_np = image_tensor.cpu().numpy()
    if img_np.ndim == 3:
        if img_np.shape[0] == 3:
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
            img_np = img_np * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            img_gray = np.mean(img_np, axis=2).astype(np.uint8)
        else:
            img_np = img_np[0]
            img_gray = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        img_gray = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_gray, mode="L").convert("RGB")
    draw    = ImageDraw.Draw(img_pil)

    if len(gt_xywh) > 0:
        gt = np.asarray(gt_xywh, dtype=np.float32)
        for row in gt:
            x1, y1 = float(row[0]), float(row[1])
            x2, y2 = x1 + float(row[2]), y1 + float(row[3])
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=1)

    if len(det_boxes) > 0:
        for row in det_boxes:
            x1, y1, x2, y2, score = row[:5]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
            if show_score:
                draw.text((x1, max(y1 - 10, 0)), f"{score:.2f}", fill=(255, 0, 0))

    img_pil.save(out_path)


def _save_detection_result(
    image_tensor,
    det_boxes: np.ndarray,
    out_path:  str,
    letterbox_info: dict = None,
) -> None:
    """
    Save detection result image with red boxes (no GT, no scores).

    Parameters
    ----------
    image_tensor : (3, H, W) or (1, H, W) or (H, W) float tensor, ImageNet normalized
    det_boxes    : (M, 5) float array [x1,y1,x2,y2,score]
    out_path     : output PNG path
    letterbox_info : dict with orig_h, orig_w, scale, dx, dy for removing letterbox padding
    """
    if not HAS_PIL:
        return

    img_np = image_tensor.cpu().numpy()
    
    if img_np.ndim == 3:
        if img_np.shape[0] == 3:
            IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_np = img_np * IMAGENET_STD + IMAGENET_MEAN
            img_np = img_np * 255.0
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            img_gray = np.mean(img_np, axis=2).astype(np.uint8)
        else:
            img_np = img_np[0]
            img_gray = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        img_gray = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    
    if letterbox_info is not None:
        orig_h = letterbox_info["orig_h"]
        orig_w = letterbox_info["orig_w"]
        scale = letterbox_info["scale"]
        dx = letterbox_info["dx"]
        dy = letterbox_info["dy"]
        
        scaled_h = int(orig_h * scale)
        scaled_w = int(orig_w * scale)
        
        img_gray = img_gray[dy:dy + scaled_h, dx:dx + scaled_w]
        
        if scale != 1.0:
            img_pil_temp = Image.fromarray(img_gray, mode="L")
            img_pil_temp = img_pil_temp.resize((orig_w, orig_h), Image.BICUBIC)
            img_gray = np.array(img_pil_temp)
        
        if len(det_boxes) > 0:
            det_boxes = det_boxes.copy()
            det_boxes[:, [0, 2]] = (det_boxes[:, [0, 2]] - dx) / scale
            det_boxes[:, [1, 3]] = (det_boxes[:, [1, 3]] - dy) / scale
            det_boxes[:, :4] = np.clip(det_boxes[:, :4], 0, [orig_w, orig_h, orig_w, orig_h])
    
    img_pil = Image.fromarray(img_gray, mode="L").convert("RGB")
    draw    = ImageDraw.Draw(img_pil)

    if len(det_boxes) > 0:
        for row in det_boxes:
            x1, y1, x2, y2 = row[:4]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)

    img_pil.save(out_path)


# ===========================================================================
# Pretty-print helper
# ===========================================================================

def print_evaluation_report(
    results: Dict[str, float],
    title: str = "Detection Evaluation Report",
) -> None:
    """
    Pretty-print a results dict returned by evaluate_detection().

    Parameters
    ----------
    results : dict with keys recall, precision, f1, ap50, fps,
              avg_inference_time_ms, total_images, total_gt, tp, fp, fn.
    title   : header string.
    """
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Recall':<25s}: {results.get('recall',    0.0) * 100:7.2f} %")
    print(f"  {'Precision':<25s}: {results.get('precision', 0.0) * 100:7.2f} %")
    print(f"  {'F1 Score':<25s}: {results.get('f1',        0.0) * 100:7.2f} %")
    print(f"  {'AP@0.5':<25s}: {results.get('ap50',       0.0) * 100:7.2f} %")
    print(sep)
    print(f"  {'FPS':<25s}: {results.get('fps',                   0.0):7.2f}")
    print(f"  {'Avg inference (ms)':<25s}: {results.get('avg_inference_time_ms', 0.0):7.2f}")
    print(sep)
    total_gt  = results.get('total_gt',     0)
    tp        = results.get('tp',           0)
    fp        = results.get('fp',           0)
    fn        = results.get('fn',           0)
    total_img = results.get('total_images', 0)
    print(f"  {'Total images':<25s}: {total_img:>7d}")
    print(f"  {'Total GT targets':<25s}: {total_gt:>7d}")
    print(f"  {'True  Positives (TP)':<25s}: {tp:>7d}")
    print(f"  {'False Positives (FP)':<25s}: {fp:>7d}")
    print(f"  {'False Negatives (FN)':<25s}: {fn:>7d}")
    print(f"{sep}\n")

    
