"""
HookManager - registers forward hooks on vision encoder, bridge module,
and language backbone layers to capture hidden-state tensors.

Supports both LLaVA-1.5 and BLIP-2/InstructBLIP-style model layouts.
Includes fallbacks for PEFT/LoRA-wrapped models.
"""

import logging
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class HookManager:
    """
    Registers forward hooks on specified submodules and collects output tensors.

    Usage:
        hm = HookManager(model, hook_spec, arch="llava")
        with hm.capture():
            outputs = model(**inputs)
        reps = hm.get_representations()
        hm.clear()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        hook_spec: Dict[str, Any],
        arch: str = "llava",
        pool: str = "mean",  # "mean" | "cls" | "last"
    ):
        self.model = model
        self.spec = hook_spec
        self.arch = arch
        self.pool = pool
        self._storage: Dict[str, Dict[int, List[torch.Tensor]]] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    # -- Path resolution -------------------------------------------------------
    def _path_aliases(self, path: str) -> List[str]:
        """
        Return candidate paths for known architecture/implementation variants.

        Handles:
        - PEFT/LoRA wrappers: base_model.model.*
        - HF LLaVA projector naming variants:
          model.multi_modal_projector vs model.mm_projector
        """
        candidates: List[str] = []

        def add(p: str) -> None:
            if p and p not in candidates:
                candidates.append(p)

        add(path)

        # Common PEFT wrapper prefix.
        add("base_model.model." + path)

        # LLaVA projector aliases across implementations.
        alias_pairs = [
            ("model.mm_projector", "model.multi_modal_projector"),
            ("model.multi_modal_projector", "model.mm_projector"),
        ]

        for old, new in alias_pairs:
            if old in path:
                aliased = path.replace(old, new)
                add(aliased)
                add("base_model.model." + aliased)

        return candidates

    def _traverse_path(self, path: str) -> Optional[torch.nn.Module]:
        """Traverse dotted path through attributes / ModuleList indices."""
        mod: Any = self.model
        for part in path.split("."):
            if part.isdigit():
                try:
                    mod = mod[int(part)]
                except (IndexError, TypeError, KeyError):
                    return None
            else:
                mod = getattr(mod, part, None)
            if mod is None:
                return None
        return mod if isinstance(mod, torch.nn.Module) else None

    def _get_submodule(self, path: str) -> Optional[Tuple[str, torch.nn.Module]]:
        """
        Resolve a path to a submodule.

        Search order:
        1. Direct path.
        2. Known aliases and PEFT prefixes.
        3. Suffix search in named_modules().
        """
        for candidate in self._path_aliases(path):
            mod = self._traverse_path(candidate)
            if mod is not None:
                return candidate, mod

        # Last-resort suffix search. This catches PEFT names such as:
        # base_model.model.model.language_model.model.layers.0
        module_map = dict(self.model.named_modules())
        for candidate in self._path_aliases(path):
            for name, module in module_map.items():
                if name == candidate or name.endswith("." + candidate):
                    return name, module

        # Extra relaxed suffix: match only the tail if full path differs.
        # Useful when HF changes one container name but layer tail is stable.
        tail = ".".join(path.split(".")[-4:])
        if tail:
            for name, module in module_map.items():
                if name.endswith("." + tail):
                    return name, module

        return None

    # -- Hook creation ---------------------------------------------------------
    def _make_hook(self, component: str, layer_key: int, pool: str) -> Callable:
        def hook_fn(module, inputs, output):
            # output may be a tuple/list, HF model output, or tensor
            if isinstance(output, (tuple, list)):
                h = output[0]
            elif hasattr(output, "last_hidden_state"):
                h = output.last_hidden_state
            elif hasattr(output, "hidden_states") and output.hidden_states is not None:
                h = output.hidden_states[-1]
            else:
                h = output

            if not isinstance(h, torch.Tensor):
                logger.debug(
                    "Skipping hook output for %s/%s because output is %s",
                    component,
                    layer_key,
                    type(h),
                )
                return

            # h shape: (batch, seq_len, hidden) or (batch, hidden)
            if h.dim() == 3:
                if pool == "mean":
                    h = h.mean(dim=1)
                elif pool == "cls":
                    h = h[:, 0, :]
                elif pool == "last":
                    h = h[:, -1, :]
            elif h.dim() > 3:
                # Vision tensors may occasionally have extra spatial dims.
                h = h.flatten(start_dim=1)

            h = h.detach().float().cpu()

            self._storage.setdefault(component, {})
            self._storage[component].setdefault(layer_key, [])
            self._storage[component][layer_key].append(h)

        return hook_fn

    def register(self) -> None:
        """Register all hooks defined in hook_spec."""
        self.remove()
        self._storage.clear()

        for component, spec in self.spec.items():
            pattern = spec["pattern"]
            sample_layers = spec["sample_layers"]

            for pos, layer_idx in enumerate(sample_layers):
                # layer_idx can be int (ModuleList index) or str (named attribute).
                full_path = f"{pattern}.{layer_idx}"
                resolved = self._get_submodule(full_path)

                # If bridge linear_1/linear_2 naming fails, try whole bridge module.
                if resolved is None and isinstance(layer_idx, str):
                    resolved = self._get_submodule(pattern)

                if resolved is None:
                    logger.warning("[%s] Could not find submodule: %s", self.arch, full_path)
                    continue

                resolved_name, mod = resolved

                # Stable integer key. Avoid Python hash() because it changes
                # between processes unless PYTHONHASHSEED is fixed.
                layer_key = int(layer_idx) if isinstance(layer_idx, int) else pos

                handle = mod.register_forward_hook(
                    self._make_hook(component, layer_key, self.pool)
                )
                self._handles.append(handle)
                logger.debug(
                    "[%s] Hooked %s as component=%s layer_key=%s",
                    self.arch,
                    resolved_name,
                    component,
                    layer_key,
                )

        logger.info(
            "[%s] Registered %d hooks across %d components",
            self.arch,
            len(self._handles),
            len(self.spec),
        )

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def capture(self):
        """Context manager: register hooks, yield, then remove."""
        self.register()
        try:
            yield self
        finally:
            self.remove()

    # -- Storage access --------------------------------------------------------
    def clear(self) -> None:
        self._storage.clear()

    def get_representations(self) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Return dict[component][layer_key] -> tensor (N, hidden_dim).
        """
        result: Dict[str, Dict[int, torch.Tensor]] = {}
        for component, layers in self._storage.items():
            result[component] = {}
            for layer_key, batches in layers.items():
                if len(batches) == 0:
                    continue
                result[component][layer_key] = torch.cat(batches, dim=0)
        return result

    def get_flat(self) -> Dict[Tuple[str, int], torch.Tensor]:
        """
        Return dict[(component, layer_key)] -> tensor (N, hidden_dim).
        """
        reps = self.get_representations()
        return {
            (component, layer_key): tensor
            for component, layers in reps.items()
            for layer_key, tensor in layers.items()
        }


# -- Batch extraction helper ---------------------------------------------------
@torch.no_grad()
def extract_representations(
    bundle: Dict[str, Any],
    dataloader: torch.utils.data.DataLoader,
    hook_spec: Dict[str, Any],
    device: str = "cuda",
    max_batches: Optional[int] = None,
) -> Tuple[Dict[str, Dict[int, torch.Tensor]], List[str]]:
    """
    Run the model over dataloader with hooks active and collect hidden states.

    Returns:
        reps_dict: dict[component][layer_key] -> tensor (N, hidden_dim)
        entity_ids_ordered: list of N entity labels aligned row-wise with reps_dict tensors

    Note:
        For correlation with RecoveryTester, we prefer entity_name because recovery
        scores are keyed by entity_name. If entity_name is absent, entity_id is used.
    """
    model = bundle["model"]
    arch = bundle["arch"]

    hm = HookManager(model, hook_spec, arch=arch)
    entity_ids_ordered: List[str] = []

    model.eval()
    with hm.capture():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            # Keep an entity key aligned with each row in the batch.
            # Prefer entity_name because RecoveryTester.run_on_forget_set uses entity_name as key.
            batch_eids = batch.get("entity_name", batch.get("entity_id", []))
            if isinstance(batch_eids, torch.Tensor):
                batch_eids = [str(x) for x in batch_eids.tolist()]
            elif isinstance(batch_eids, (str, int)):
                batch_eids = [str(batch_eids)]
            else:
                batch_eids = [str(x) for x in batch_eids]

            batch_tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            try:
                if arch == "llava":
                    model(
                        input_ids=batch_tensors.get("input_ids"),
                        attention_mask=batch_tensors.get("attention_mask"),
                        pixel_values=batch_tensors.get("pixel_values"),
                    )
                elif arch == "blip2":
                    model(
                        pixel_values=batch_tensors.get("pixel_values"),
                        input_ids=batch_tensors.get("input_ids"),
                        attention_mask=batch_tensors.get("attention_mask"),
                    )
                else:
                    model(**batch_tensors)

                # Add entity labels only after successful forward pass.
                entity_ids_ordered.extend(batch_eids)

            except Exception as e:
                logger.warning("Forward pass error at batch %d: %s", i, e)
                continue

    return hm.get_representations(), entity_ids_ordered
