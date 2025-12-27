#
# engine.py
# mammography-pipelines
#
# Training/validation utilities for the density models, including Grad-CAM, metrics, and embedding extraction.
#
# Thales Matheus Mendonça Santos - November 2025
#
import os
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    scaler: Optional[GradScaler] = None,
    amp_enabled: bool = False,
) -> Tuple[float, float]:
    """One epoch of supervised training with optional AMP and extra tabular features."""
    model.train()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0
    
    logger = logging.getLogger("mammography")
    log_per_iter = logger.isEnabledFor(logging.DEBUG)
    last_step_end = time.perf_counter()

    for step, batch in enumerate(tqdm(loader, desc="Train", leave=False), 1):
        iter_start = time.perf_counter()
        data_wait = iter_start - last_step_end
        
        if len(batch) == 4:
            x, y, _, extra_features = batch
        else:
            x, y, _ = batch
            extra_features = None

        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        mask = y_tensor >= 0
        if not mask.any():
            continue

        x = x[mask]
        y_tensor = y_tensor[mask]

        extra_tensor = None
        if extra_features is not None:
            extra_tensor = extra_features.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Use autocast+GradScaler when AMP is enabled to keep training stable.
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, extra_tensor)
            loss = loss_fn(logits, y_tensor)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        iter_total = time.perf_counter() - iter_start
        last_step_end = time.perf_counter()

        if log_per_iter:
            logger.debug(f"iter={step} | wait={data_wait*1e3:.2f}ms | total={iter_total*1e3:.2f}ms")

        losses.append(loss.item())
        pred = logits.detach().float().argmax(dim=1)
        correct += (pred == y_tensor).sum().item()
        total += y_tensor.numel()
        
    acc = correct / max(1, total)
    return float(np.mean(losses) if losses else 0.0), acc

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    loss_fn: Optional[nn.Module] = None,
    collect_preds: bool = False,
    gradcam: bool = False,
    gradcam_dir: Optional[Path] = None,
    gradcam_limit: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validation loop that returns metrics plus optional per-sample predictions/Grad-CAMs."""
    model.eval()
    all_y = []
    all_p = []
    all_prob = []
    rows = []
    losses: List[float] = []
    gradcam_saved = 0
    pred_rows: List[Dict[str, Any]] = []

    for batch in tqdm(loader, desc="Val", leave=False):
        if len(batch) == 4:
            x, y, metas, extra_features = batch
        else:
            x, y, metas = batch
            extra_features = None

        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)
        mask = y_tensor >= 0
        if not mask.any():
            continue
        x = x[mask]
        y_tensor = y_tensor[mask]
        metas = [m for idx, m in enumerate(metas) if mask[idx]]
        extra_tensor = None
        if extra_features is not None:
            extra_tensor = extra_features.to(device=device, non_blocking=True)
            extra_tensor = extra_tensor[mask]

        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, extra_tensor)
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, y_tensor).item()))

        logits = logits.float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()

        all_prob.append(probs)
        all_p.append(pred)
        all_y.append(y_tensor.cpu().numpy())
        rows.extend(list(metas))

        if collect_preds:
            for i, m in enumerate(metas):
                row = {
                    "path": m.get("path"),
                    "accession": m.get("accession"),
                    "raw_label": m.get("raw_label"),
                    "y_true": int(y_tensor[i].item()),
                    "y_pred": int(pred[i]),
                    "probs": probs[i].tolist(),
                }
                pred_rows.append(row)
        if gradcam and gradcam_dir is not None and gradcam_saved < gradcam_limit:
            gradcam_saved += _save_gradcam_batch(model, x, torch.tensor(pred, device=device), metas, gradcam_dir, gradcam_saved, device)

    if not all_prob:
        return {"acc": 0.0, "loss": float(np.mean(losses)) if losses else 0.0}, pred_rows

    y_true = np.concatenate(all_y).astype(int)
    y_pred = np.concatenate(all_p).astype(int)
    prob = np.concatenate(all_prob, axis=0)

    num_classes = prob.shape[1]

    # Label adjustment for metrics (assuming 4 classes = 1..4, 2 classes = 0..1)
    if num_classes == 4:
        y_true_mapped = y_true + 1
        y_pred_mapped = y_pred + 1
        labels = [1, 2, 3, 4]
    else:
        y_true_mapped = y_true
        y_pred_mapped = y_pred
        labels = list(range(num_classes))

    acc = accuracy_score(y_true_mapped, y_pred_mapped)

    kappa_q = 0.0
    if num_classes > 1:
        try:
            kappa_q = cohen_kappa_score(y_true_mapped, y_pred_mapped, weights='quadratic')
        except Exception:
            pass

    auc_ovr = None
    if num_classes > 1:
        try:
            if num_classes == 2:
                auc_ovr = roc_auc_score(y_true, prob[:, 1])
            else:
                auc_ovr = roc_auc_score(pd.get_dummies(y_true), prob, multi_class='ovr')
        except Exception:
            pass

    cm_np = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels)
    cm = cm_np.tolist()
    report = classification_report(y_true_mapped, y_pred_mapped, labels=labels, output_dict=True, zero_division=0)
    macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = np.diag(cm_np) / cm_np.sum(axis=1)
    recalls = np.nan_to_num(recalls, nan=0.0)
    bal_acc = float(np.mean(recalls)) if num_classes > 0 else None
    chance = 1.0 / float(num_classes) if num_classes > 1 else 0.0
    bal_acc_adj = None
    if bal_acc is not None and num_classes > 1:
        denom = 1.0 - chance
        bal_acc_adj = float((bal_acc - chance) / denom) if denom > 0 else 0.0

    out_rows = []
    for i, m in enumerate(rows):
        pr = prob[i]
        row_data = {
            **m,
            "y_true": int(y_true_mapped[i]),
            "y_pred": int(y_pred_mapped[i]),
        }
        for c_idx in range(num_classes):
            row_data[f"p_class_{c_idx}"] = float(pr[c_idx])
        out_rows.append(row_data)

    metrics = {
        "acc": acc,
        "kappa_quadratic": kappa_q,
        "auc_ovr": auc_ovr,
        "macro_f1": macro_f1,
        "bal_acc": bal_acc,
        "bal_acc_adj": bal_acc_adj,
        "confusion_matrix": cm,
        "classification_report": report,
        "loss": float(np.mean(losses)) if losses else None,
        "val_rows": out_rows,
    }
    return metrics, pred_rows

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
    layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Register a forward hook to capture pooled CNN embeddings for each batch."""
    model.eval()
    feats: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    buffer: List[np.ndarray] = []

    def hook_fn(module, inp, out):
        v = out.detach().float().cpu().numpy()
        v = v.reshape(v.shape[0], -1)
        buffer.append(v)

    # Try to register a hook on the requested layer (by name) or default avgpool.
    target_layer = None
    if layer_name:
        target_layer = dict(model.named_modules()).get(layer_name)
        if target_layer is None:
            logging.getLogger("mammography").warning("Layer %s nao encontrado; usando avgpool.", layer_name)
    if target_layer is None:
        if hasattr(model, "avgpool"):
            target_layer = model.avgpool
        elif hasattr(model, "backbone") and hasattr(model.backbone, "avgpool"):
            target_layer = model.backbone.avgpool

    handle = None
    if target_layer:
        handle = target_layer.register_forward_hook(hook_fn)

    for batch in tqdm(loader, desc="Embeddings", leave=False):
        if len(batch) == 4:
            x, _, metas, extra_features = batch
        else:
            x, _, metas = batch
            extra_features = None
            
        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        extra_tensor = None
        if extra_features is not None:
            extra_tensor = extra_features.to(device=device, non_blocking=True)
            
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            _ = model(x, extra_tensor)
            
        if buffer:
            feat = np.concatenate(buffer, axis=0)
            buffer.clear()
            feats.append(feat)
        rows.extend(list(metas))

    if handle is not None:
        handle.remove()

    if not feats:
        return np.zeros((0, 0)), []

    return np.concatenate(feats, axis=0), rows


def _save_gradcam_batch(
    model: nn.Module,
    x: torch.Tensor,
    preds: torch.Tensor,
    metas: List[Dict[str, Any]],
    out_dir: Path,
    already: int,
    device: torch.device,
) -> int:
    """Generate Grad-CAM heatmaps for a small batch and persist blended overlays."""
    try:
        target_layer = None
        if hasattr(model, "layer4"):
            target_layer = model.layer4[-1]
        elif hasattr(model, "features"):
            target_layer = model.features[-1]
        if target_layer is None:
            return 0

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def fwd_hook(_, __, output):
            activations.append(output.detach())

        def bwd_hook(_, grad_in, grad_out):
            gradients.append(grad_out[0].detach())

        handle_fwd = target_layer.register_forward_hook(fwd_hook)
        handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

        out = model(x)
        class_idxs = preds.detach()
        selected = out.gather(1, class_idxs.unsqueeze(1)).sum()
        model.zero_grad()
        selected.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        if not activations or not gradients:
            return 0
        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        cam = (cam - cam.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / (cam.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-6)

        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for i in range(min(len(metas), cam.shape[0])):
            heatmap = cam[i].detach().cpu().numpy()
            img = x[i].detach().cpu()
            img = img.permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = Image.fromarray(heatmap).resize((img.shape[1], img.shape[0]))
            heatmap_img = heatmap_img.convert("RGBA")
            base = Image.fromarray(np.uint8(img * 255))
            base = base.convert("RGBA")
            blended = Image.blend(base, heatmap_img, alpha=0.35)
            fname = out_dir / f"gradcam_{already + saved}_{metas[i].get('accession','sample')}.png"
            blended.save(fname)
            saved += 1
        return saved
    except Exception as exc:
        logging.getLogger("mammography").warning("Grad-CAM falhou: %s", exc)
        return 0


def plot_history(history: List[Dict[str, Any]], outdir: Path) -> None:
    """Persist training curves to CSV/PNG so notebooks and LaTeX can consume them."""
    if not history:
        return
    df = pd.DataFrame(history)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "train_history.csv", index=False)
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(df["epoch"], df["train_loss"], label="train")
        ax[0].plot(df["epoch"], df["val_loss"], label="val")
        ax[0].set_title("Loss")
        ax[0].legend()
        ax[1].plot(df["epoch"], df["train_acc"], label="train")
        ax[1].plot(df["epoch"], df["val_acc"], label="val")
        ax[1].set_title("Accuracy")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(outdir / "train_history.png", dpi=150)
        plt.close(fig)
    except Exception:
        logging.getLogger("mammography").debug("Plot de history falhou; salvando apenas CSV.")


def save_predictions(pred_rows: List[Dict[str, Any]], outdir: Path) -> None:
    """Write per-sample validation predictions when the caller opts in."""
    if not pred_rows:
        return
    pd.DataFrame(pred_rows).to_csv(outdir / "val_predictions.csv", index=False)

def save_metrics_figure(metrics: Dict[str, Any], out_path: str) -> None:
    """Render confusion matrix and per-class precision/recall/F1 to a single PNG."""
    try:
        cm_data = np.array(metrics.get("confusion_matrix", []), dtype=float)
        report = metrics.get("classification_report", {})
        
        if cm_data.size == 0:
            return

        classes = [str(k) for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Matrix
        ax = axes[0]
        cax = ax.imshow(cm_data, interpolation='nearest', cmap=mpl_cm.Blues)
        ax.set_title('Confusion Matrix')
        fig.colorbar(cax, ax=ax)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        
        thresh = cm_data.max() / 2.
        for i, j in np.ndindex(cm_data.shape):
            ax.text(j, i, format(int(cm_data[i, j]), 'd'),
                     horizontalalignment="center",
                     color="white" if cm_data[i, j] > thresh else "black")
        
        # Metrics
        ax = axes[1]
        # Simple bar plot of precision/recall/f1 per class
        precisions = [report[c]['precision'] for c in classes]
        recalls = [report[c]['recall'] for c in classes]
        f1s = [report[c]['f1-score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision')
        ax.bar(x, recalls, width, label='Recall')
        ax.bar(x + width, f1s, width, label='F1')
        
        ax.set_ylabel('Score')
        ax.set_title('Metrics per Class')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        logging.getLogger("mammography").warning(f"Failed to save metrics figure: {e}")
