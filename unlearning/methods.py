"""
Unlearning method implementations.

Baselines:
  1. GA           - Gradient Ascent on forget set
  2. NPO          - Negative Preference Optimization
  3. MMUnlearner  - Visual Pattern Erasure (visual contrastive loss)
  4. MANU         - Modality-Aware Neuron Pruning
  5. CAGUL        - Cross-Modal Attention Guided Unlearning
  6. SineProject  - Projector Stability Unlearning
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from config import UnlearningConfig, CHECKPOINT_ROOT

logger = logging.getLogger(__name__)


# -- Shared training utilities -------------------------------------------------
def _get_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)


def _move_batch(batch: Dict, device: str) -> Dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# -- 1. Gradient Ascent (GA) ---------------------------------------------------
def unlearn_ga(
    bundle     : Dict[str, Any],
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    cfg        : UnlearningConfig,
    device     : str = "cuda",
    save_path  : Optional[str] = None,
) -> nn.Module:
    """
    Maximise CE loss on forget set while minimising it on retain set.
    Loss = -L_forget + alpha * L_retain
    """
    model = bundle["model"]
    model.train()
    opt   = _get_optimizer(model, cfg.forget_lr)

    retain_iter = iter(retain_loader)
    step = 0

    logger.info("GA unlearning started")
    while step < cfg.num_steps:
        for forget_batch in forget_loader:
            if step >= cfg.num_steps:
                break

            forget_batch = _move_batch(forget_batch, device)

            # Forget loss (ascent)
            out_f = model(**forget_batch)
            l_forget = out_f.loss if hasattr(out_f, "loss") else out_f[0]

            # Retain loss
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = _move_batch(retain_batch, device)

            out_r = model(**retain_batch)
            l_retain = out_r.loss if hasattr(out_r, "loss") else out_r[0]

            loss = -l_forget + cfg.alpha * l_retain

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1 == 0:
                logger.info(
                    f"[GA] step={step+1}/{cfg.num_steps}  "
                    f"forget={l_forget.item():.4f}  "
                    f"retain={l_retain.item():.4f}"
                )
            step += 1

    if save_path:
        model.save_pretrained(save_path)
    return model


# -- 2. Negative Preference Optimization (NPO) --------------------------------
def unlearn_npo(
    bundle       : Dict[str, Any],
    forget_loader: DataLoader,
    retain_loader: DataLoader,
    ref_bundle   : Dict[str, Any],
    cfg          : UnlearningConfig,
    device       : str = "cuda",
    save_path    : Optional[str] = None,
) -> nn.Module:
    """
    Treat forget set as "rejected" examples (negative DPO style).
    Loss = -log σ(-β * (log p_θ(f) - log p_ref(f))) + α * L_retain
    """
    model     = bundle["model"]
    ref_model = ref_bundle["model"]

    model.train()
    ref_model.eval()
    opt = _get_optimizer(model, cfg.forget_lr)

    retain_iter = iter(retain_loader)
    step = 0

    logger.info("NPO unlearning started")
    while step < cfg.num_steps:
        for forget_batch in forget_loader:
            if step >= cfg.num_steps:
                break

            forget_batch = _move_batch(forget_batch, device)

            # Log-probs from current model
            out_θ  = model(**forget_batch)
            logits_θ = out_θ.logits  # (B, T, V)
            labels   = forget_batch.get("labels", forget_batch.get("input_ids"))

            lp_θ  = _token_avg_logprob(logits_θ, labels)

            # Log-probs from reference model (frozen)
            with torch.no_grad():
                out_ref = ref_model(**forget_batch)
            lp_ref = _token_avg_logprob(out_ref.logits, labels)

            # NPO loss: push θ away from ref on forget set
            l_npo = -F.logsigmoid(-cfg.beta * (lp_θ - lp_ref.detach())).mean()

            # Retain loss
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = _move_batch(retain_batch, device)
            out_r   = model(**retain_batch)
            l_retain = out_r.loss if hasattr(out_r, "loss") else out_r[0]

            loss = l_npo + cfg.alpha * l_retain

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1 == 0:
                logger.info(
                    f"[NPO] step={step+1}/{cfg.num_steps}  "
                    f"npo={l_npo.item():.4f}  retain={l_retain.item():.4f}"
                )
            step += 1

    if save_path:
        model.save_pretrained(save_path)
    return model


def _token_avg_logprob(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean log probability of the target tokens per sample."""
    B, T, V = logits.shape
    lp = F.log_softmax(logits[:, :-1], dim=-1)          # (B, T-1, V)
    tgt = labels[:, 1:].clamp(0)                         # (B, T-1)
    # gather log-probs for target tokens
    gathered = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
    mask = (labels[:, 1:] != -100).float()
    return (gathered * mask).sum(-1) / (mask.sum(-1) + 1e-8)  # (B,)


# -- 3. MMUnlearner - Visual Pattern Erasure -----------------------------------
def unlearn_mmunlearner(
    bundle          : Dict[str, Any],
    forget_loader   : DataLoader,
    retain_loader   : DataLoader,
    cfg             : UnlearningConfig,
    device          : str = "cuda",
    save_path       : Optional[str] = None,
) -> nn.Module:
    """
    MMUnlearner: erase visual patterns for the forget concept while
    preserving textual knowledge.

    Loss = L_visual_contrastive(forget) + α * L_language(retain)
    The visual contrastive term pushes forget visual embeddings toward
    a random reference, breaking the image->concept association.
    """
    model = bundle["model"]
    model.train()
    opt = _get_optimizer(model, cfg.forget_lr)

    retain_iter = iter(retain_loader)
    step = 0

    logger.info("MMUnlearner unlearning started")
    while step < cfg.num_steps:
        for forget_batch in forget_loader:
            if step >= cfg.num_steps:
                break

            forget_batch = _move_batch(forget_batch, device)

            # Visual contrastive loss: push vision encoder output
            # of forget images toward a random unit vector
            pixel_values = forget_batch.get("pixel_values")
            if pixel_values is None:
                step += 1
                continue

            # Extract visual features
            arch = bundle["arch"]
            if arch == "llava":
                # Robust LLaVA vision tower access for HF + PEFT/LoRA wrappers
                vision_tower = None

                # Try common object paths first
                candidates = [
                    getattr(model, "vision_tower", None),
                    getattr(getattr(model, "model", None), "vision_tower", None),
                    getattr(getattr(getattr(model, "model", None), "model", None), "vision_tower", None),
                    getattr(getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None), "vision_tower", None),
                ]

                for cand in candidates:
                    if cand is not None:
                        vision_tower = cand
                        break

                # Last-resort search by module name
                if vision_tower is None:
                    for n, m in model.named_modules():
                        if n.endswith("vision_tower"):
                            vision_tower = m
                            break

                if vision_tower is None:
                    raise AttributeError("Could not locate LLaVA vision_tower in model/PEFT wrapper")

                vis_out = vision_tower(pixel_values)
                # vis_out.last_hidden_state: (B, patches, D)
                vis_feat = vis_out.last_hidden_state.mean(dim=1)  # (B, D)
            else:  # blip2
                vis_out  = model.vision_model(pixel_values)
                vis_feat = vis_out.last_hidden_state.mean(dim=1)

            # Random target (different per batch to avoid mode collapse)
            target = torch.randn_like(vis_feat)
            target = F.normalize(target, dim=-1)

            vis_feat_n  = F.normalize(vis_feat, dim=-1)
            l_visual    = (vis_feat_n * target).sum(dim=-1).mean() + 1.0
            # This is maximised similarity to random -> minimise it -> push forget visual out

            # Retain language loss
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = _move_batch(retain_batch, device)
            out_r    = model(**retain_batch)
            l_retain = out_r.loss if hasattr(out_r, "loss") else out_r[0]

            loss = l_visual + cfg.alpha * l_retain

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1 == 0:
                logger.info(
                    f"[MMUnlearner] step={step:4d}  "
                    f"visual={l_visual.item():.4f}  retain={l_retain.item():.4f}"
                )
            step += 1

    if save_path:
        model.save_pretrained(save_path)
    return model


# -- 4. MANU - Modality-Aware Neuron Pruning -----------------------------------
def unlearn_manu(
    bundle        : Dict[str, Any],
    forget_loader : DataLoader,
    retain_loader : DataLoader,
    cfg           : UnlearningConfig,
    device        : str = "cuda",
    save_path     : Optional[str] = None,
) -> nn.Module:
    """
    MANU: identify neurons most responsible for forget knowledge
    by computing activation differences (forget - retain), then
    prune (zero out) the top-k% most discriminative neurons.
    After pruning, fine-tune briefly on retain set to recover utility.
    """
    model = bundle["model"]
    model.eval()

    logger.info("MANU: computing neuron importance scores ...")

    # Collect activation differences
    neuron_scores: Dict[str, torch.Tensor] = {}
    hooks, handles = [], []

    def _make_act_hook(name: str, is_forget: bool):
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach().float().mean(dim=0).cpu()  # mean over batch
            if name not in neuron_scores:
                neuron_scores[name] = torch.zeros_like(h)
            if is_forget:
                neuron_scores[name] += h
            else:
                neuron_scores[name] -= h
        return hook_fn

    # Register hooks on all MLP intermediate activations
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "mlp" in name.lower():
            handle = mod.register_forward_hook(_make_act_hook(name, is_forget=True))
            handles.append((handle, name, True))

    # Forward pass: forget set
    with torch.no_grad():
        for batch in forget_loader:
            batch = _move_batch(batch, device)
            try:
                model(**batch)
            except Exception:
                pass

    # Switch to retain set
    for handle, name, _ in handles:
        handle.remove()
    handles.clear()

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "mlp" in name.lower():
            handle = mod.register_forward_hook(_make_act_hook(name, is_forget=False))
            handles.append((handle, name, False))

    with torch.no_grad():
        for batch in retain_loader:
            batch = _move_batch(batch, device)
            try:
                model(**batch)
            except Exception:
                pass

    for handle, _, _ in handles:
        handle.remove()

    # Prune top-k% neurons per layer
    logger.info(f"MANU: pruning top-{cfg.prune_ratio * 100:.0f}% neurons ...")
    for name, mod in model.named_modules():
        if name not in neuron_scores:
            continue

        if not hasattr(mod, "weight") or mod.weight is None:
            continue

        scores = neuron_scores[name].abs().flatten()

        with torch.no_grad():
            flat = mod.weight.data.view(-1)

            # Compute top-k locally for this layer/tensor only.
            # This avoids applying global indices to smaller tensors.
            k = max(1, int(min(scores.numel(), flat.numel()) * cfg.prune_ratio))
            k = min(k, flat.numel())

            if k <= 0:
                continue

            local_scores = scores[:flat.numel()].to(flat.device)
            top_k_idx_local = torch.topk(local_scores, k).indices

            top_k_idx_local = top_k_idx_local[
                (top_k_idx_local >= 0) & (top_k_idx_local < flat.numel())
                ]

            if top_k_idx_local.numel() > 0:
                flat[top_k_idx_local] = 0.0

    # Brief retain fine-tune to recover utility
    logger.info("MANU: retain fine-tuning ...")
    model.train()
    opt  = _get_optimizer(model, cfg.retain_lr)
    step = 0
    for retain_batch in retain_loader:
        if step >= cfg.num_steps // 5:
            break
        retain_batch = _move_batch(retain_batch, device)
        out  = model(**retain_batch)
        loss = out.loss if hasattr(out, "loss") else out[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 10 == 0:
            logger.info(f"[MANU] retain step={step}  loss={loss.item():.4f}")

    if save_path:
        model.save_pretrained(save_path)
    return model


# -- 5. CAGUL - Cross-Modal Attention Guided Unlearning -----------------------
def unlearn_cagul(
    bundle          : Dict[str, Any],
    forget_loader   : DataLoader,
    retain_loader   : DataLoader,
    cfg             : UnlearningConfig,
    device          : str = "cuda",
    save_path       : Optional[str] = None,
) -> nn.Module:
    """
    CAGUL: use cross-modal attention scores to identify which visual tokens
    are most relevant to the forget concept, then apply targeted updates
    to those tokens to break the image->concept mapping.

    For architectures without explicit cross-attention, approximates
    by using the visual tokens with highest attention weight variance.
    """
    model = bundle["model"]
    model.train()
    opt  = _get_optimizer(model, cfg.forget_lr)

    retain_iter = iter(retain_loader)
    step = 0

    logger.info("CAGUL unlearning started")
    while step < cfg.num_steps:
        for forget_batch in forget_loader:
            if step >= cfg.num_steps:
                break

            forget_batch = _move_batch(forget_batch, device)

            # Forward with attention outputs
            out_f = model(**forget_batch, output_attentions=True)
            l_forget = out_f.loss if hasattr(out_f, "loss") else out_f[0]

            # Get attention weights (last layer of LM, averaged over heads)
            attentions = getattr(out_f, "attentions", None)  # list of (B, H, T, T)
            if attentions is not None and len(attentions) > 0:
                last_attn = attentions[-1]  # (B, H, T, T)
                # Mean over heads: (B, T, T)
                avg_attn  = last_attn.mean(dim=1)
                # Identify visual token positions with highest attention
                # (approximation: tokens beyond text length)
                # cfg.attn_threshold may be float; slicing needs integer.
                # If threshold <= 1, treat it as a fraction of sequence length.
                if cfg.attn_threshold <= 1:
                    attn_cutoff = int(avg_attn.size(-1) * cfg.attn_threshold)
                else:
                    attn_cutoff = int(cfg.attn_threshold)

                attn_cutoff = max(1, min(attn_cutoff, avg_attn.size(-1)))
                attn_to_vis = avg_attn[:, :, :attn_cutoff].mean()
                # Penalise high attention to visual tokens for forget input
                l_attn = attn_to_vis
            else:
                l_attn = torch.tensor(0.0, device=device)

            # Retain loss
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = _move_batch(retain_batch, device)
            out_r    = model(**retain_batch)
            l_retain = out_r.loss if hasattr(out_r, "loss") else out_r[0]

            loss = l_forget.detach() * 0 - l_forget + l_attn + cfg.alpha * l_retain

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1 == 0:
                logger.info(
                    f"[CAGUL] step={step:4d}  "
                    f"forget={l_forget.item():.4f}  "
                    f"attn={l_attn.item():.4f}  "
                    f"retain={l_retain.item():.4f}"
                )
            step += 1

    if save_path:
        model.save_pretrained(save_path)
    return model


# -- 6. SineProject - Projector Stability Unlearning --------------------------
def unlearn_sineproject(
    bundle          : Dict[str, Any],
    forget_loader   : DataLoader,
    retain_loader   : DataLoader,
    cfg             : UnlearningConfig,
    device          : str = "cuda",
    save_path       : Optional[str] = None,
) -> nn.Module:
    """
    SineProject: unlearn while stabilising the bridge module (MLP projector
    or Q-Former) by penalising large changes to its Jacobian.

    Loss = -L_forget + α * L_retain + λ_proj * L_proj_stability
    where L_proj_stability measures weight drift of the bridge module
    relative to its pre-unlearning state.
    """
    model = bundle["model"]
    arch  = bundle["arch"]
    model.train()

    # Snapshot bridge module weights before unlearning
    bridge_snapshot = _snapshot_bridge(model, arch)

    opt  = _get_optimizer(model, cfg.forget_lr)
    retain_iter = iter(retain_loader)
    step = 0

    logger.info("SineProject unlearning started")
    while step < cfg.num_steps:
        for forget_batch in forget_loader:
            if step >= cfg.num_steps:
                break

            forget_batch = _move_batch(forget_batch, device)

            out_f    = model(**forget_batch)
            l_forget = out_f.loss if hasattr(out_f, "loss") else out_f[0]

            # Retain loss
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)
            retain_batch = _move_batch(retain_batch, device)
            out_r    = model(**retain_batch)
            l_retain = out_r.loss if hasattr(out_r, "loss") else out_r[0]

            # Bridge stability loss: penalise drift from original weights
            l_proj = _bridge_drift(model, arch, bridge_snapshot)

            loss = -l_forget + cfg.alpha * l_retain + cfg.proj_lambda * l_proj

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % 1 == 0:
                logger.info(
                    f"[SineProject] step={step:4d}  "
                    f"forget={l_forget.item():.4f}  "
                    f"retain={l_retain.item():.4f}  "
                    f"proj_drift={l_proj.item():.4f}"
                )
            step += 1

    if save_path:
        model.save_pretrained(save_path)
    return model


def _snapshot_bridge(model: nn.Module, arch: str) -> Dict[str, torch.Tensor]:
    snapshot = {}
    prefixes = ["mm_projector", "multi_modal_projector"] if arch == "llava" else ["qformer"]

    for name, param in model.named_parameters():
        if any(p in name for p in prefixes):
            snapshot[name] = param.data.detach().cpu().clone()

    return snapshot


def _bridge_drift(
    model: nn.Module,
    arch: str,
    snapshot: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """
    L2 norm of parameter drift in the bridge module since unlearning started.
    """
    prefixes = ["mm_projector", "multi_modal_projector"] if arch == "llava" else ["qformer"]
    drift  = torch.tensor(0.0, device=next(model.parameters()).device)
    count  = 0
    for name, param in model.named_parameters():
        if name in snapshot:
            orig  = snapshot[name].to(param.device)
            drift = drift + (param - orig).pow(2).sum()
            count += param.numel()
    return drift / (count + 1e-8)


# -- Dispatcher ----------------------------------------------------------------
def run_unlearning(
    method          : str,
    bundle          : Dict[str, Any],
    forget_loader   : DataLoader,
    retain_loader   : DataLoader,
    cfg             : UnlearningConfig,
    ref_bundle      : Optional[Dict[str, Any]] = None,
    device          : str = "cuda",
    save_path       : Optional[str] = None,
) -> nn.Module:
    """Dispatch to the correct unlearning method by name."""

    logger.info(f"Running unlearning method: {method}")

    if method == "ga":
        return unlearn_ga(bundle, forget_loader, retain_loader, cfg, device, save_path)

    elif method == "npo":
        if ref_bundle is None:
            raise ValueError("NPO requires a reference model bundle (ref_bundle)")
        return unlearn_npo(
            bundle, forget_loader, retain_loader, ref_bundle, cfg, device, save_path
        )

    elif method == "mmunlearner":
        return unlearn_mmunlearner(
            bundle, forget_loader, retain_loader, cfg, device, save_path
        )

    elif method == "manu":
        return unlearn_manu(
            bundle, forget_loader, retain_loader, cfg, device, save_path
        )

    elif method == "cagul":
        return unlearn_cagul(
            bundle, forget_loader, retain_loader, cfg, device, save_path
        )

    elif method == "sineproject":
        return unlearn_sineproject(
            bundle, forget_loader, retain_loader, cfg, device, save_path
        )

    else:
        raise ValueError(f"Unknown unlearning method: {method}")






