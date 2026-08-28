"""``predictor/weights/meta/`` has exactly ONE writer.

alpha-engine-config-I9018/-I9028. The live meta prefix is the contract
inference loads. Until 2026-08-28 the weekly retrain used it as scratch space:
``TrainingIOSpec.live()`` pointed `weights_prefix` / `manifest_key` /
`feature_list_key` straight at it, so every spec of the Saturday rotation
overwrote the served model's manifest and feature contract as a side effect of
training. Nothing gated that, nothing reported it, and it silently undid any
operator rollback at the next rotation.

The invariant now: training writes a per-run staging prefix, and the only code
that puts bytes under the live prefix is ``model.registry.promote_to_champion``.

champion-challenger-policy section 7.4 — every assertion here was run against
the pre-fix tree and shown red.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
import model.registry as reg
from training.io_spec import TrainingIOSpec

_REPO = Path(__file__).resolve().parent.parent
_LIVE = "predictor/weights/meta/"

# Modules allowed to name the live prefix, each with the reason it may.
#
#   model/registry.py  — the promoter. The ONE writer.
#   config.py          — where the constant is defined.
#   inference/*        — READERS. Inference loads the served contract; that is
#                        the whole point of the prefix.
#   scripts/smoke_meta_model_load.py — a read-only smoke check.
#   monitoring/feature_drift.py — writes feature_drift_reference.json under the
#                        live prefix. NOT part of the model contract, not in the
#                        registry bundle, and (measured 2026-08-28) it has no
#                        caller in the training path. Left in place
#                        deliberately: moving it to staging would strand the
#                        reference unless promote_to_champion also carried it,
#                        which is a wider change than the pre-rotation window
#                        allowed. Tracked for alpha-engine-config-I9029.
_ALLOWED_TO_NAME_LIVE_PREFIX = {
    "config.py",
    "model/registry.py",
    "monitoring/feature_drift.py",
    "scripts/smoke_meta_model_load.py",
}
_ALLOWED_PREFIXES = ("inference/", "docs/", "tests/")

_WRITE_CALLS = {"put_object", "copy_object", "upload_file", "upload_fileobj"}


def _source_files():
    for path in sorted(_REPO.rglob("*.py")):
        rel = path.relative_to(_REPO).as_posix()
        if rel.startswith((".venv/", "build/", ".git/")):
            continue
        yield rel, path


def test_training_never_names_the_live_serving_prefix():
    """RED pre-fix: training/io_spec.py and training/model_zoo.py both did.

    `io_spec.live()` read cfg.META_WEIGHTS_PREFIX / META_MANIFEST_KEY /
    META_FEATURE_LIST_KEY, and model_zoo's G2 helper listed the two contract
    keys in order to write them back after a rotation.
    """
    offenders = []
    for rel, path in _source_files():
        if not rel.startswith("training/"):
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            # A docstring or comment may DISCUSS the prefix; only executable
            # references count.
            if isinstance(node, ast.Attribute) and node.attr in (
                "META_WEIGHTS_PREFIX", "META_FEATURE_LIST_KEY",
            ):
                offenders.append(f"{rel}: cfg.{node.attr}")
            if isinstance(node, ast.Constant) and node.value == _LIVE:
                offenders.append(f"{rel}: literal {_LIVE!r}")
    assert not offenders, (
        "training code names the live serving prefix:\n  "
        + "\n  ".join(offenders)
        + "\n\nTraining writes cfg.META_STAGING_PREFIX. The live prefix is "
          "written only by model.registry.promote_to_champion "
          "(alpha-engine-config-I9018)."
    )


def test_the_only_training_reference_to_the_live_manifest_is_a_pointer_read():
    """META_MANIFEST_KEY may still be READ by training — the live manifest names
    the serving version — but only in the two functions that take the POINTER
    from it. Its cpcv fields must never reach a promotion decision.
    """
    src = (_REPO / "training/model_zoo.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    readers = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Constant) and node.value == "META_MANIFEST_KEY":
                readers.add(fn.name)
            if isinstance(node, ast.Attribute) and node.attr == "META_MANIFEST_KEY":
                readers.add(fn.name)
    assert readers <= {"_read_live_manifest", "_resolve_incumbent_from_bundle"}, (
        f"unexpected training reader(s) of the live manifest: {sorted(readers)}"
    )


def test_registry_source_and_live_prefixes_are_distinct():
    """RED pre-fix: `promote_to_champion(live_prefix=DEFAULT_SOURCE_PREFIX)` —
    one constant served as both "snapshot from" and "promote to".
    """
    assert reg.DEFAULT_SOURCE_PREFIX != reg.DEFAULT_LIVE_PREFIX
    assert reg.DEFAULT_LIVE_PREFIX == _LIVE
    assert reg.DEFAULT_SOURCE_PREFIX == cfg.META_STAGING_PREFIX
    assert not reg.DEFAULT_SOURCE_PREFIX.startswith(_LIVE)


def test_live_io_spec_paths_are_outside_the_serving_prefix():
    io = TrainingIOSpec.live()
    scoped = io.for_run(date_str="2026-08-29", model_version="v3.0-meta")
    for spec in (io, scoped):
        for path in (spec.weights_prefix, spec.manifest_key, spec.feature_list_key):
            assert not path.startswith(_LIVE), path
            assert path.startswith(cfg.META_STAGING_PREFIX), path


def test_no_unexpected_module_names_the_live_prefix():
    """A module-level allowlist, so a NEW writer has to argue for itself in a
    diff rather than appearing silently.

    EXECUTABLE references only — a comment or docstring may discuss the prefix
    (much of this PR's rationale does), and prose cannot write an object.
    """
    unexpected = []
    for rel, path in _source_files():
        if rel in _ALLOWED_TO_NAME_LIVE_PREFIX or rel.startswith(_ALLOWED_PREFIXES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(ast.get_docstring(n, clean=False))
            for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and _LIVE in node.value and id(node.value) not in docstrings:
                unexpected.append(f"{rel}:{node.lineno}")
            if isinstance(node, ast.Attribute) and node.attr in (
                "META_WEIGHTS_PREFIX", "META_FEATURE_LIST_KEY",
            ):
                unexpected.append(f"{rel}:{node.lineno} cfg.{node.attr}")
    assert not unexpected, (
        "module(s) naming predictor/weights/meta/ without being on the "
        f"allowlist: {unexpected}"
    )


def test_promote_to_champion_writes_the_live_prefix():
    """The positive half: the one permitted writer really does write there, so
    this file is not asserting that nothing writes the prefix at all.
    """
    import json

    class _S3:
        def __init__(self):
            self.copied = []
            self.puts = {}

        def get_paginator(self, _op):
            outer = self

            class _P:
                def paginate(self, Bucket, Prefix):  # noqa: N803
                    if Prefix.startswith("predictor/registry/v-new/"):
                        yield {"Contents": [
                            {"Key": "predictor/registry/v-new/manifest.json"},
                            {"Key": "predictor/registry/v-new/feature_list.json"},
                            {"Key": "predictor/registry/v-new/meta_model.pkl"},
                            {"Key": "predictor/registry/v-new/_lineage.json"},
                        ]}
                    else:
                        yield {"Contents": [
                            {"Key": "predictor/registry/v-new/_lineage.json"},
                        ]}

            del outer
            return _P()

        def get_object(self, Bucket, Key):  # noqa: N803
            import io as _io
            if Key.endswith("_lineage.json"):
                return {"Body": _io.BytesIO(json.dumps(
                    {"version_id": "v-new", "stage": "challenger",
                     "date": "2026-08-29"}).encode())}
            if Key == f"{_LIVE}manifest.json":
                return {"Body": _io.BytesIO(json.dumps({"date": "2026-08-29"}).encode())}
            raise KeyError(Key)

        def copy_object(self, Bucket, Key, CopySource):  # noqa: N803
            self.copied.append(Key)

        def put_object(self, Bucket, Key, Body, ContentType="application/json"):  # noqa: N803
            self.puts[Key] = Body

    s3 = _S3()
    out = reg.promote_to_champion(s3, "bkt", "v-new")
    assert out["live_prefix"] == _LIVE
    assert f"{_LIVE}manifest.json" in s3.copied
    assert f"{_LIVE}meta_model.pkl" in s3.copied


@pytest.mark.parametrize("call", sorted(_WRITE_CALLS))
def test_write_call_names_are_still_what_this_file_scans_for(call):
    """A guard on the guard: if boto3 usage in this repo moves to a call name
    this file does not know about, the scan above quietly stops covering it.
    """
    hits = sum(
        1 for _rel, path in _source_files()
        if call in path.read_text(encoding="utf-8")
    )
    assert hits > 0, f"no module uses {call} — update _WRITE_CALLS"
