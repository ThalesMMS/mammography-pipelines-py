#!/usr/bin/env python3
# ruff: noqa
#
# train_modeling.py
# mammography-pipelines
#
# Model construction and optimization helpers for density training.
# DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
# It must NOT be used for clinical or medical diagnostic purposes.
# No medical decision should be based on these results.
#
# Thales Matheus Mendonça Santos - November 2025
#
"""Modeling helpers for mammography density training."""
from typing import Optional
import torch

from mammography.utils.common import (
    parse_float_list,
)

def _parse_class_weights(raw: str, num_classes: int):
    """
    Parse a command-line class-weights specification into a canonical form.
    
    Parameters:
        raw (str): Raw input string from CLI describing class weights (e.g., "none", "auto", "[0.1,0.9]", "0.1,0.9").
        num_classes (int): Expected number of classes when a numeric list is provided; used to validate parsed list length.
    
    Returns:
        Union[str, list[float]]: The string `"none"` or `"auto"` if specified or inferred, or a list of `num_classes` floats representing class weights.
    
    Raises:
        SystemExit: If the numeric parsing/validation fails; the exit message contains the parse error.
    """
    text = str(raw or "").strip()
    if not text:
        return "none"
    lower = text.lower()
    if lower in {"none", "auto"}:
        return lower
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    try:
        weights = parse_float_list(text, expected_len=num_classes, name="class_weights")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return weights or "none"

def _resolve_head_module(model: torch.nn.Module, arch: str) -> Optional[torch.nn.Module]:
    """
    Locate and return the model's classification head module based on common architecture attribute names.
    
    Parameters:
        arch (str): Architecture identifier used for special-case resolution (e.g., "resnet50" to prefer `fc`).
        
    Returns:
        torch.nn.Module or None: The module that serves as the classification head if found, `None` otherwise.
    """
    if hasattr(model, "classifier"):
        return model.classifier
    if arch == "resnet50" and hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "heads"):
        return model.heads
    if hasattr(model, "head"):
        return model.head
    return None

def _resolve_backbone_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the model's backbone module if present, otherwise return the model itself.
    
    Parameters:
        model (torch.nn.Module): Model which may expose a `backbone` attribute.
    
    Returns:
        torch.nn.Module: The `model.backbone` module when available, otherwise `model`.
    """
    return model.backbone if hasattr(model, "backbone") else model

def freeze_backbone(model: torch.nn.Module, arch: str) -> None:
    """
    Freeze all model parameters except the classification head so only the head remains trainable.
    
    This sets `requires_grad = False` for every parameter in `model`, then resolves the model's classification head using the provided `arch` string and sets `requires_grad = True` for all parameters in that head if found.
    
    Parameters:
        model (torch.nn.Module): The model whose backbone/head parameter gradients will be modified.
        arch (str): Architecture identifier used to locate the classifier head (e.g., "resnet50", "efficientnet_b0", "vit").
    """
    for p in model.parameters():
        p.requires_grad = False
    head = _resolve_head_module(model, arch)
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True

def unfreeze_last_block(model: torch.nn.Module, arch: str) -> None:
    """
    Unfreezes (sets requires_grad=True) the final encoder/block of the model backbone for architecture-specific backbones.
    
    This enables gradient updates only for the last training block of the backbone so that downstream training can fine-tune the final stage while earlier backbone stages remain frozen. Behavior by `arch`:
    - "resnet50": unfreezes `backbone.layer4` if present.
    - "efficientnet_b0": unfreezes the last module in `backbone.features` if present.
    - "vit", "vit_b_16", "vit_b_32", "vit_l_16": unfreezes the last entry of `backbone.encoder.layers` if present.
    - "deit_small", "deit_base": unfreezes the last entry of `backbone.blocks` if present.
    
    Parameters:
        model (torch.nn.Module): Model whose backbone will be modified.
        arch (str): Architecture identifier used to select which backbone block to unfreeze.
    """
    backbone = _resolve_backbone_module(model)
    if arch == "resnet50" and hasattr(backbone, "layer4"):
        for p in backbone.layer4.parameters():
            p.requires_grad = True
    if arch == "efficientnet_b0" and hasattr(backbone, "features"):
        features = backbone.features
        if len(features) > 0:
            for p in features[-1].parameters():
                p.requires_grad = True
    if arch in {"vit", "vit_b_16", "vit_b_32", "vit_l_16"}:
        encoder = getattr(backbone, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is not None and len(layers) > 0:
            for p in layers[-1].parameters():
                p.requires_grad = True
    if arch in {"deit_small", "deit_base"}:
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            for p in blocks[-1].parameters():
                p.requires_grad = True

def build_param_groups(model: torch.nn.Module, arch: str, lr_head: float, lr_backbone: float) -> list[dict]:
    """
    Create optimizer parameter groups that separate classifier head parameters from backbone parameters and assign distinct learning rates.
    
    If the model's head module can be resolved, parameters are split by membership in that head; otherwise parameters are partitioned using architecture-specific name prefixes (e.g., "fc" for resnet50, "classifier" for others). Only parameters with requires_grad=True are included.
    
    Parameters:
        model (torch.nn.Module): The model containing backbone and classification head.
        arch (str): Architecture identifier used to choose naming conventions when a head module cannot be resolved.
        lr_head (float): Learning rate to assign to the head parameter group.
        lr_backbone (float): Learning rate to assign to the backbone parameter group.
    
    Returns:
        list[dict]: A list of optimizer parameter-group dictionaries. Each dict contains a "params" entry with parameter iterables and an "lr" entry set to the corresponding learning rate; groups are included only if they contain parameters.
    """
    head = _resolve_head_module(model, arch)
    if head is None:
        if arch == "resnet50":
            head_params = [p for n, p in model.named_parameters() if n.startswith("fc") and p.requires_grad]
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc") and p.requires_grad]
        else:
            head_params = [p for n, p in model.named_parameters() if n.startswith("classifier") and p.requires_grad]
            backbone_params = [p for n, p in model.named_parameters() if not n.startswith("classifier") and p.requires_grad]
    else:
        head_ids = {id(p) for p in head.parameters()}
        head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    params = []
    if head_params:
        params.append({"params": head_params, "lr": lr_head})
    if backbone_params:
        params.append({"params": backbone_params, "lr": lr_backbone})
    return params
