"""Opt-in single generation slot for one managed llama.cpp model child.

``local_runtime.generation_slots`` left null does not write a slot count.
llama.cpp b11671 treats a missing ``--parallel`` as automatic: ``n_parallel < 0``
sets four slots and turns unified KV on. The value ``1`` writes ``parallel = 1``.
On a model that was on that automatic path it also writes ``kv-unified = on``,
because an explicit slot count skips the branch that enables unified KV.

A model that already pins ``parallel`` (the DFlash draft, which is one slot
for its own recurrent-state copies) keeps that pin and does not gain a
``kv-unified`` key.

The value is read only from the machine root ``config.yaml``
(``get_default_hermes_root()``). The managed server, its presets, and its
models live there. A profile ``config.yaml`` is a separate file: profiles do
not inherit this key, and a caller profile, a context-local home override, or
a worker snapshot cannot set it. Cold start and the owner's idle sweep use
that same root file, including a Curator turn scoped to a profile.

The router argv is not part of this setting. ``local_runtime.extra_args`` is
not read. Nothing here queues an HTTP request or loads a model.
"""

from __future__ import annotations

SINGLE_GENERATION_SLOT = 1


class GenerationSlotsError(ValueError):
    """``local_runtime.generation_slots`` is not null and not ``1``."""


class MachineConfigUnreadable(OSError):
    """The machine-root config file exists but could not be parsed."""


def machine_generation_slots() -> int | None:
    """Slot count from the machine-root config only.

    Missing file or missing key is null (automatic slots). A value other than
    null or ``1`` raises ``GenerationSlotsError``. An unreadable file raises
    ``MachineConfigUnreadable``. The active profile, ``HERMES_HOME`` when it
    points at ``profiles/<name>``, and any config dict a caller passed in are
    not consulted.
    """
    from hermes_constants import get_default_hermes_root
    from hermes_cli.config_backend import read_config_doc_readonly

    path = get_default_hermes_root() / "config.yaml"
    try:
        data = read_config_doc_readonly(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MachineConfigUnreadable(str(exc)) from exc
    except Exception as exc:
        raise MachineConfigUnreadable(str(exc)) from exc
    if data is None:
        return None
    if not isinstance(data, dict):
        raise MachineConfigUnreadable("machine-root config.yaml is not a mapping")
    section = data.get("local_runtime")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise MachineConfigUnreadable("machine-root local_runtime is not a mapping")
    if "generation_slots" not in section:
        return None
    return resolve_generation_slots(section.get("generation_slots"))


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
