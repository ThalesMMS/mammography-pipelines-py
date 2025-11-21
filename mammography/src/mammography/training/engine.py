import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
        
        extra_tensor = None
        if extra_features is not None:
            extra_tensor = extra_features.to(device=device, non_blocking=True)
            
        optimizer.zero_grad(set_to_none=True)
        
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
def validate(model: nn.Module, loader: DataLoader, device: torch.device, amp_enabled: bool = False) -> Dict[str, Any]:
    model.eval()
    all_y = []
    all_p = []
    all_prob = []
    rows = []
    
    for batch in tqdm(loader, desc="Val", leave=False):
        if len(batch) == 4:
            x, y, metas, extra_features = batch
        else:
            x, y, metas = batch
            extra_features = None
            
        x = x.to(device=device, non_blocking=True, memory_format=torch.channels_last)
        extra_tensor = None
        if extra_features is not None:
            extra_tensor = extra_features.to(device=device, non_blocking=True)
            
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(x, extra_tensor)
            
        logits = logits.float()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(dim=1).cpu().numpy()
        
        all_prob.append(probs)
        all_p.append(pred)
        all_y.append(np.array(y))
        rows.extend(list(metas))
        
    y_true = np.concatenate(all_y).astype(int)
    y_pred = np.concatenate(all_p).astype(int)
    prob = np.concatenate(all_prob, axis=0)
    
    num_classes = prob.shape[1]
    
    # Ajuste de labels para métricas (assumindo 4 classes = 1..4, 2 classes = 0..1)
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
        except:
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
            
    cm = confusion_matrix(y_true_mapped, y_pred_mapped, labels=labels).tolist()
    report = classification_report(y_true_mapped, y_pred_mapped, labels=labels, output_dict=True, zero_division=0)
    
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
        
    return {
        "acc": acc,
        "kappa_quadratic": kappa_q,
        "auc_ovr": auc_ovr,
        "confusion_matrix": cm,
        "classification_report": report,
        "val_rows": out_rows,
    }

@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    feats: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    buffer: List[np.ndarray] = []

    def hook_fn(module, inp, out):
        v = out.detach().float().cpu().numpy()
        v = v.reshape(v.shape[0], -1)
        buffer.append(v)

    # Tenta registrar hook na avgpool (EfficientNet/ResNet)
    target_layer = None
    if hasattr(model, "avgpool"):
        target_layer = model.avgpool
    elif hasattr(model, "backbone") and hasattr(model.backbone, "avgpool"):
         target_layer = model.backbone.avgpool
    
    if target_layer is None:
        # Fallback genérico se não achar avgpool
        pass

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

def save_metrics_figure(metrics: Dict[str, Any], out_path: str) -> None:
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
