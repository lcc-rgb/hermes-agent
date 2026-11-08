"""Opt-in single generation slot: preset, router argv, and config propagation.

These tests build launch keys and argv in-process. They do not start
llama-server and do not show that a Qwen process queued a second request.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.local_runtime.generation_slots import (
    GenerationSlotsError, apply_generation_slots, resolve_generation_slots)
from tests.hermes_cli.test_local_external_draft import BUILD, ENTRY, ROOMY

_PRESERVED = (
    "ctx-size", "batch-size", "ubatch-size", "cache-type-k", "cache-type-v",
    "flash-attn", "spec-type", "spec-draft-n-max", "model",
    "backend-sampling", "spec-draft-backend-sampling",
)


def test_default_generation_slots_is_unset():
    assert DEFAULT_CONFIG["local_runtime"]["generation_slots"] is None


@pytest.mark.parametrize("value", [None])
def test_null_generation_slots_resolve_to_automatic(value):
    assert resolve_generation_slots(value) is None


@pytest.mark.parametrize("value", [0, 2, 4, -1, True, False, "1", 1.0, [], {}])
def test_invalid_generation_slots_are_rejected(value):
    with pytest.raises(GenerationSlotsError):
        resolve_generation_slots(value)
    keys = {"ctx-size": "65536", "cache-type-k": "q8_0"}
    with pytest.raises(GenerationSlotsError):
        apply_generation_slots(keys, value if isinstance(value, int) and not isinstance(value, bool) else 4)
    assert keys == {"ctx-size": "65536", "cache-type-k": "q8_0"}


def _stage(tmp_path, monkeypatch, *, draft_on_disk: bool):
    from hermes_cli.local_runtime import bootstrap, catalog, presets

    monkeypatch.setattr(catalog, "entry_for_model",
                        lambda model_id: ENTRY if model_id == BUILD.model_id else None)
    monkeypatch.setattr(presets, "read_gguf_header", lambda _p: SimpleNamespace(sampling_defaults={}))
    monkeypatch.setattr(presets, "profile_from_gguf", lambda _h: replace(
        ENTRY.profile(BUILD), kv_scale=1.0, recurrent_state_bytes=150 << 20))
    draft = bootstrap.assets_dir() / ENTRY.draft.local_name
    draft.parent.mkdir(parents=True, exist_ok=True)
    if draft_on_disk:
        draft.touch()
    elif draft.exists():
        draft.unlink()
    return tmp_path / f"{BUILD.model_id}.gguf"


def _mtp_preset(tmp_path, monkeypatch, slots):
    from hermes_cli.local_runtime import presets

    gguf = _stage(tmp_path, monkeypatch, draft_on_disk=False)
    return presets.preset_for_model(gguf, ROOMY, set(), generation_slots=slots)


def test_mtp_preset_gains_one_slot_and_keeps_unified_kv(tmp_path, monkeypatch):
    """An MTP model with no parallel key is the automatic four-slot path.

    Opting in adds parallel and kv-unified and leaves the planner's other
    launch keys alone. kv-unified is required because llama.cpp b11671 turns
    unified KV on only while the slot count is still automatic.
    """
    from hermes_cli.local_runtime.presets import render_presets

    automatic = _mtp_preset(tmp_path, monkeypatch, None)
    single = _mtp_preset(tmp_path, monkeypatch, 1)

    assert automatic.keys.get("spec-type") == "draft-mtp"
    assert "parallel" not in automatic.keys
    assert "kv-unified" not in automatic.keys
    assert single.window == automatic.window
    assert single.spilled == automatic.spilled
    assert set(single.keys) - set(automatic.keys) == {"parallel", "kv-unified"}
    for key in _PRESERVED:
        assert single.keys[key] == automatic.keys[key]
    assert single.keys["parallel"] == "1"
    assert single.keys["kv-unified"] == "on"
    assert single.keys["cache-type-k"] == "q8_0"
    assert single.keys["cache-type-v"] == "q8_0"
    rendered = render_presets([single])
    assert rendered.count("parallel = 1") == 1
    assert f"ctx-size = {single.window}" in rendered


def test_dflash_one_slot_pin_is_unchanged_by_the_opt_in(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import presets

    gguf = _stage(tmp_path, monkeypatch, draft_on_disk=True)
    automatic = presets.preset_for_model(gguf, ROOMY, set(), engine_tag="b11671")
    single = presets.preset_for_model(
        gguf, ROOMY, set(), engine_tag="b11671", generation_slots=1)

    assert automatic.keys["spec-type"] == "draft-dflash"
    assert automatic.keys["parallel"] == "1"
    assert "kv-unified" not in automatic.keys
    assert single.keys == automatic.keys
    assert single.window == automatic.window


def test_two_synthetic_callers_share_one_preset_and_no_queue(tmp_path, monkeypatch):
    """Both callers get the same preset. Nothing in that preset is a queue.

    What this proves: the opt-in does not give either caller a private
    context, cache, spec type, or a second slot. What it does not prove:
    llama.cpp's deferral order, a wait bound, an HTTP busy status, or that
    a real Qwen generation was queued. Those happen inside the engine
    process, which this test does not start.
    """
    single = _mtp_preset(tmp_path, monkeypatch, 1)
    issued = {name: dict(single.keys) for name in ("synthetic-a", "synthetic-b")}

    assert issued["synthetic-a"] == issued["synthetic-b"]
    assert issued["synthetic-a"]["parallel"] == "1"
    assert "queue" not in issued["synthetic-a"]
    assert not any(key.startswith("caller") for key in issued["synthetic-a"])


def test_router_argv_does_not_carry_the_slot_pin(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import supervisor as sup_mod

    monkeypatch.setattr(sup_mod, "_direct_io_args", lambda _exe: ())
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"")
    preset = tmp_path / "presets.ini"
    preset.write_text(
        "[synthetic]\nctx-size = 65536\nparallel = 1\nkv-unified = on\n",
        encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    bare = sup_mod.LlamaServerSupervisor(
        binary, models, preset_path=preset, port=9, extra_args=[])
    argv = bare.router_argv()

    assert "--models-preset" in argv
    assert argv[argv.index("--models-preset") + 1] == str(preset)
    assert "--parallel" not in argv
    assert "--kv-unified" not in argv
    assert "--ctx-size" not in argv
    assert "--cache-type-k" not in argv

    with_programmatic = sup_mod.LlamaServerSupervisor(
        binary, models, preset_path=preset, port=9, extra_args=["--parallel", "1"])
    assert with_programmatic.router_argv()[-2:] == ["--parallel", "1"]


def _root_config(home, text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_boot_reads_the_machine_root_and_drops_yaml_extra_args(tmp_path, monkeypatch):
    """The caller's generation_slots is not the launch value. Root is."""
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _root_config(home, "local_runtime:\n  generation_slots: 1\n")
    from hermes_cli.local_runtime import binaries, bootstrap
    from hermes_cli.local_runtime import endpoint

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    monkeypatch.setattr(bootstrap, "_rejected_generation_slots_token", None)
    models = bootstrap.models_dir()
    models.mkdir(parents=True, exist_ok=True)
    (models / "synthetic.gguf").touch()
    monkeypatch.setattr(bootstrap, "_detect_gpu_vendor", lambda: None)
    monkeypatch.setattr(bootstrap, "_admitted_models_max", lambda *_a, **_k: 1)
    monkeypatch.setattr(endpoint, "_state_endpoint", lambda: None)
    monkeypatch.setattr(binaries, "installed_engine",
                        lambda backend="auto", **_k: binaries.Engine(
                            "cpu", "b1", tmp_path / "llama-server"))
    seen: dict = {}

    def _generate(*_a, **kwargs):
        seen["generation_slots"] = kwargs.get("generation_slots")
        path = tmp_path / "presets.ini"
        path.write_text("", encoding="utf-8")
        return path

    class _Supervisor:
        def __init__(self, *_a, **kwargs):
            seen["supervisor"] = kwargs
            self.base_url = "http://127.0.0.1:9/v1"
            self.proc = None

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(bootstrap, "_generate_presets", _generate)
    monkeypatch.setattr(
        "hermes_cli.local_runtime.supervisor.LlamaServerSupervisor", _Supervisor)
    try:
        started = bootstrap.ensure_local_runtime({
            "local_runtime": {
                "enabled": True,
                "generation_slots": 4,
                "extra_args": ["--parallel", "1", "--ctx-size", "1"],
            },
        })
    finally:
        bootstrap._SUPERVISOR = None
        bootstrap._SERVING_ENGINE = None

    assert started is not None
    assert seen["generation_slots"] == 1
    assert "extra_args" not in seen["supervisor"]


def test_boot_refuses_an_invalid_root_slot_count_before_spawn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    _root_config(home, "local_runtime:\n  generation_slots: 4\n")
    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_SUPERVISOR", None)
    monkeypatch.setattr(bootstrap, "_rejected_generation_slots_token", None)
    constructed: list = []
    monkeypatch.setattr(
        "hermes_cli.local_runtime.supervisor.LlamaServerSupervisor",
        lambda *_a, **_k: constructed.append(1))

    assert bootstrap.ensure_local_runtime(
        {"local_runtime": {"enabled": True, "generation_slots": 1}}) is None
    assert constructed == []


def test_profile_config_cannot_override_or_hide_the_root_slot_count(tmp_path, monkeypatch):
    """A profile file and a caller dict disagree with the machine root.

    Curator and other profile-scoped turns load their own config. The shared
    server still uses the root key, and the idle reader uses that same key.
    """
    from hermes_constants import (
        get_hermes_home, reset_hermes_home_override, set_hermes_home_override)
    from hermes_cli.config import load_config_readonly
    from hermes_cli.local_runtime.generation_slots import machine_generation_slots

    root = tmp_path / "root"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text(
        "local_runtime:\n  generation_slots: 1\n", encoding="utf-8")
    (profile / "config.yaml").write_text(
        "local_runtime:\n  generation_slots: 4\nmodel:\n  provider: custom\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    token = set_hermes_home_override(profile)
    try:
        assert get_hermes_home() == profile
        profile_value = (load_config_readonly().get("local_runtime") or {}).get("generation_slots")
        assert profile_value == 4
        assert machine_generation_slots() == 1
        assert get_hermes_home() == profile
    finally:
        reset_hermes_home_override(token)

    from hermes_cli.local_runtime import bootstrap

    monkeypatch.setattr(bootstrap, "_rejected_generation_slots_token", None)
    slots, ok = bootstrap._running_generation_slots()
    assert ok is True and slots == 1


class _Router:
    def __init__(self, preset_path):
        self.preset_path = preset_path
        self.models_max = 1
        self.proc = SimpleNamespace(poll=lambda: None)
        self._lifecycle_lock = threading.RLock()
        self._refit_usable = None
        self.reloads = 0
        self.status = "unloaded"

    def models(self, timeout_s=30):
        return {"synthetic": self.status}

    def reload_presets(self):
        self.reloads += 1


def _refit_ready(tmp_path, monkeypatch):
    """Staged MTP model, hardware probes pinned, presets written with automatic slots."""
    from hermes_cli.local_runtime import bootstrap, presets

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(bootstrap, "_rejected_generation_slots_token", None)
    monkeypatch.setattr(bootstrap, "_SERVING_ENGINE", None)
    gguf = _stage(tmp_path, monkeypatch, draft_on_disk=False)
    mdir = tmp_path / "models"
    mdir.mkdir()
    staged = mdir / gguf.name
    staged.touch()
    monkeypatch.setattr(bootstrap, "models_dir", lambda: mdir)
    monkeypatch.setattr(bootstrap, "_launch_budget", lambda *_a, **_k: ROOMY)
    from hermes_cli.local_runtime import hardware
    monkeypatch.setattr(hardware, "probe_budget", lambda **_k: ROOMY)
    path = tmp_path / "presets.ini"
    presets.generate_presets(mdir, ROOMY, path)
    return path


def _write_slots(tmp_path, value: str):
    (tmp_path / "home" / "config.yaml").write_text(
        f"local_runtime:\n  generation_slots: {value}\n", encoding="utf-8")


def test_idle_refit_applies_and_rolls_back_the_config_file(tmp_path, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, presets

    path = _refit_ready(tmp_path, monkeypatch)
    before = presets.read_preset_decisions(path)
    model_id = next(iter(before))
    router = _Router(path)

    _write_slots(tmp_path, "1")
    assert bootstrap.refit_idle_presets(router) is True
    assert router.reloads == 1
    after = presets.read_preset_decisions(path)[model_id]
    assert after.keys["parallel"] == "1"
    assert after.keys["kv-unified"] == "on"
    for key in _PRESERVED:
        assert after.keys[key] == before[model_id].keys[key]
    assert after.window == before[model_id].window

    assert bootstrap.refit_idle_presets(router) is False
    assert router.reloads == 1

    _write_slots(tmp_path, "4")
    kept = path.read_text(encoding="utf-8")
    assert bootstrap.refit_idle_presets(router) is False
    assert path.read_text(encoding="utf-8") == kept
    assert router.reloads == 1

    router.status = "loaded"
    _write_slots(tmp_path, "null")
    assert bootstrap.refit_idle_presets(router) is False
    assert path.read_text(encoding="utf-8") == kept

    router.status = "unloaded"
    assert bootstrap.refit_idle_presets(router) is True
    rolled = presets.read_preset_decisions(path)[model_id]
    assert "parallel" not in rolled.keys
    assert "kv-unified" not in rolled.keys
    for key in _PRESERVED:
        assert rolled.keys[key] == before[model_id].keys[key]
    assert router.reloads == 2
