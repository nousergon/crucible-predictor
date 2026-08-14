"""Shared support for the ``bootstrap_spot()`` tests.

``bootstrap_spot()`` in ``infrastructure/_spot_common.sh`` no longer carries
an inline heredoc — it constructs a ``krepis.spot_bootstrap render`` CLI
invocation and hands the result to ``run_ssm`` (alpha-engine-config-I4992/
I6922 cutover; the deletion tests/test_spot_bootstrap_invariants.py's own
docstring called for). The four ``test_spot_bootstrap_*`` modules beside this
one used to regex the heredoc text directly; they now need the same two
things this module provides:

1. the raw CLI invocation, so a test can assert something about how the
   *launcher* builds its arguments (env-closure class: is every value the
   renderer needs actually guaranteed non-empty before it runs?);
2. the RENDERED script, produced by feeding those arguments through the real
   ``krepis.spot_bootstrap.render_bootstrap`` — the same function
   ``bootstrap_spot()`` invokes at runtime — so a test can assert something
   about the script krepis actually emits, not a restated copy of it.

Not a test module itself (no ``test_`` prefix) — pytest does not collect it.
"""

from __future__ import annotations

import re
from pathlib import Path

from krepis.spot_bootstrap import ConfigCopy, SpotBootstrapSpec, render_bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
SPOT_COMMON = REPO_ROOT / "infrastructure" / "_spot_common.sh"

_FLAG_RE = re.compile(
    r'--([\w-]+)\s+(?:"([^"]*)"|\'([^\']*)\'|(\S+))'
)


def spot_common_text() -> str:
    return SPOT_COMMON.read_text(encoding="utf-8")


def bootstrap_spot_block(text: str | None = None) -> str:
    """The body of ``bootstrap_spot()``."""
    text = text if text is not None else spot_common_text()
    m = re.search(r"\nbootstrap_spot\(\)\s*\{(.*?)\n\}", text, re.S)
    assert m, f"bootstrap_spot() not found in {SPOT_COMMON.name}"
    return m.group(1)


def render_invocation(block: str | None = None) -> str:
    """The raw text of the ``krepis.spot_bootstrap render ...`` call."""
    block = block if block is not None else bootstrap_spot_block()
    m = re.search(r"krepis\.spot_bootstrap\s+render\s+(.*?)\)\"", block, re.S)
    assert m, (
        "no `krepis.spot_bootstrap render` invocation found in bootstrap_spot() "
        "— has the cutover (alpha-engine-config-I4992/I6922) been reverted?"
    )
    return m.group(1)


def render_args(invocation: str | None = None) -> list[tuple[str, str]]:
    """[(flag, raw_value), ...] — raw_value keeps any ``$VAR``/``${VAR}`` text."""
    invocation = invocation if invocation is not None else render_invocation()
    cleaned = invocation.replace("\\\n", " ")
    out: list[tuple[str, str]] = []
    for m in _FLAG_RE.finditer(cleaned):
        flag = m.group(1)
        raw = next(g for g in m.groups()[1:] if g is not None)
        out.append((flag, raw))
    return out


#: Shell variables the render call may reference whose value is only known at
#: SSM-dispatch runtime (assigned by ``spot_launch()`` before ``bootstrap_spot``
#: is ever called), plus a stand-in used ONLY for producing a diffable render —
#: never asserted as the real value.
_RUNTIME_ONLY_STANDINS = {
    "_S3_STAGING": "s3://alpha-engine-research/tmp/spot_train/RUNID",
}

#: MAX_RUNTIME_SECONDS has a static top-of-file default in _spot_common.sh
#: of "" (deliberately empty — see that file's "Per-stage identity" comment:
#: it must be a per-stage assignment, never defaulted here, so `set -u` is
#: satisfied at source time while the real per-stage value stays load-
#: bearing). `_static_default` would therefore resolve it to the empty
#: string, which is a legitimate STATIC answer but not what the renderer
#: ever actually sees: `spot_launch()` asserts it non-empty before any
#: instance exists, and bootstrap_spot() only ever runs after spot_launch()
#: has populated `_INSTANCE_ID`/`_S3_STAGING` — so by the time the render
#: call executes, MAX_RUNTIME_SECONDS is guaranteed real, exactly like
#: `_S3_STAGING` above. Stand in with a representative value so
#: `rendered_script()` actually exercises the --max-runtime-seconds path.
_RUNTIME_ONLY_STANDINS["MAX_RUNTIME_SECONDS"] = "5400"


def _static_default(name: str, text: str) -> str | None:
    """A literal or ``${NAME:-default}``-declared value for a global var.

    Returns ``None`` if ``name`` has no static (launcher-source-visible)
    default — the caller must fall back to a runtime-only stand-in or fail.
    """
    m = re.search(rf'^{re.escape(name)}="([^"$]*)"\s*$', text, re.M)
    if m:
        return m.group(1)
    m = re.search(rf'^{re.escape(name)}="\$\{{{re.escape(name)}:-([^}}"]*)\}}"\s*$', text, re.M)
    if m:
        return m.group(1)
    return None


def resolve(raw: str, text: str | None = None) -> str:
    """Expand every ``$VAR``/``${VAR}``/``${VAR:-default}`` in ``raw``.

    Resolution order: the render call's OWN inline default (``${BRANCH:-main}``)
    wins first (it is what actually executes); otherwise the variable's static
    top-of-file default; otherwise a known runtime-only stand-in. Anything
    else is a genuine env-closure gap and raises, which is the failure mode
    this module exists to make loud rather than silently substitute for.
    """
    text = text if text is not None else spot_common_text()

    def _sub_braced(m: re.Match) -> str:
        name, tail = m.group(1), m.group(2)
        if tail.startswith(":-") or tail.startswith("-"):
            return tail[2:] if tail.startswith(":-") else tail[1:]
        if name in _RUNTIME_ONLY_STANDINS:
            return _RUNTIME_ONLY_STANDINS[name]
        default = _static_default(name, text)
        if default is not None:
            return default
        raise AssertionError(
            f"render invocation references ${{{name}}}, which has no static "
            f"default in {SPOT_COMMON.name} and is not a declared runtime-only "
            f"variable — the launcher cannot guarantee it is set before "
            f"bootstrap_spot() constructs the render call."
        )

    def _sub_bare(m: re.Match) -> str:
        name = m.group(1)
        if name in _RUNTIME_ONLY_STANDINS:
            return _RUNTIME_ONLY_STANDINS[name]
        default = _static_default(name, text)
        if default is not None:
            return default
        raise AssertionError(
            f"render invocation references ${name}, which has no static "
            f"default in {SPOT_COMMON.name} and is not a declared runtime-only "
            f"variable."
        )

    out = re.sub(r"\$\{(\w+)([^}]*)\}", _sub_braced, raw)
    out = re.sub(r"\$(\w+)\b", _sub_bare, out)
    return out


def resolved_args(text: str | None = None) -> dict[str, list[str]]:
    """flag -> [resolved values] (repeatable flags keep every occurrence)."""
    text = text if text is not None else spot_common_text()
    out: dict[str, list[str]] = {}
    for flag, raw in render_args(render_invocation(bootstrap_spot_block(text))):
        out.setdefault(flag, []).append(resolve(raw, text))
    return out


def spec_from_resolved(args: dict[str, list[str]]) -> SpotBootstrapSpec:
    exports: dict[str, str] = {}
    for kv in args.get("export", []):
        key, value = kv.split("=", 1)
        exports[key] = value
    copies: list[ConfigCopy] = []
    for raw in args.get("config-copy", []):
        parts = raw.split(":")
        copies.append(ConfigCopy(source_name=parts[0], dest=parts[1]))
    max_runtime_raw = args.get("max-runtime-seconds", [None])[0]
    return SpotBootstrapSpec(
        repo_url=args["repo-url"][0],
        checkout=args["checkout"][0],
        config_copies=tuple(copies),
        exports=exports,
        branch=args.get("branch", ["main"])[0],
        region=args.get("region", ["us-east-1"])[0],
        max_runtime_seconds=int(max_runtime_raw) if max_runtime_raw else None,
    )


def rendered_script(text: str | None = None) -> str:
    """The exact script ``bootstrap_spot()`` sends to ``run_ssm`` today.

    A real call through ``krepis.spot_bootstrap.render_bootstrap`` — not a
    restated copy — using the arguments extracted (and resolved) from the
    live source. This is what every ``test_spot_bootstrap_*`` module should
    assert against now: the RENDERED output, never the deleted heredoc.
    """
    text = text if text is not None else spot_common_text()
    return render_bootstrap(spec_from_resolved(resolved_args(text)))
