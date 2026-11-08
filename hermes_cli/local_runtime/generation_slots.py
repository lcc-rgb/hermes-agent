"""Opt-in single generation slot for one managed llama.cpp model child.

``local_runtime.generation_slots`` left null does not write a slot count.
llama.cpp b11671 treats a missing ``--parallel`` as automatic: ``n_parallel < 0``
sets four slots and turns unified KV on. The value ``1`` writes ``parallel = 1``.
On a model that was on that automatic path it also writes ``kv-unified = on``,
because an explicit slot count skips the branch that enables unified KV.

A model that already pins ``parallel`` (the DFlash draft, which is one slot
for its own recurrent-state copies) keeps that pin and does not gain a
``kv-unified`` key.

The router argv is not part of this setting. ``local_runtime.extra_args`` is
not read. Nothing here queues an HTTP request or loads a model.
"""

from __future__ import annotations

SINGLE_GENERATION_SLOT = 1


class GenerationSlotsError(ValueError):
    """``local_runtime.generation_slots`` is not null and not ``1``."""


def resolve_generation_slots(value: object) -> int | None:
    """``None`` leaves the engine's automatic slot count. ``1`` is the only opt-in."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value != SINGLE_GENERATION_SLOT:
        raise GenerationSlotsError(
            "local_runtime.generation_slots accepts only 1 or null, "
            f"got {value!r}")
    return SINGLE_GENERATION_SLOT


def apply_generation_slots(keys: dict[str, str], slots: int | None) -> None:
    """Pin one generation slot onto a preset key dict. ``None`` changes nothing."""
    if slots is None:
        return
    if slots != SINGLE_GENERATION_SLOT:
        raise GenerationSlotsError(
            "local_runtime.generation_slots accepts only 1 or null, "
            f"got {slots!r}")
    # b11671 enables unified KV only inside the automatic-slot branch.
    # Pinning parallel by itself would leave kv_unified at the struct default.
    on_auto_path = "parallel" not in keys
    keys["parallel"] = "1"
    if on_auto_path:
        keys["kv-unified"] = "on"
