"""alpha-engine-config-I12351: a weekly-SF rehearsal never writes regime/latest.json.

RegimeSubstrate and RegimeRetrospectiveEval run with ``action: produce`` on a
rehearsal too. Under the rehearsal marker (the execution name in the Task
Payload's ``execution_arn``) every S3 write is re-keyed under
``rehearsal/<name>/``; on a production execution the keys are unchanged.

Real boto3 clients over an in-memory transport registered at ``before-send``:
that event fires after the isolation hook has rewritten the parameters and the
request has been serialized, so the URL the fake records is exactly the key
that would have reached S3. (botocore's Stubber asserts parameters at
``before-parameter-build.*.*``, which runs BEFORE a service-specific handler
and so cannot see the rewrite.)
"""

from __future__ import annotations

import io
import boto3
import pytest
from urllib.parse import unquote, urlsplit

from botocore.awsrequest import AWSResponse

import regime.handler as regime_handler
import regime.retrospective_eval_handler as retro_handler
from regime import rehearsal
from regime.rehearsal import isolate, namespace_prefix, rehearsal_name, tolerate_stale
from regime.substrate import write_regime_substrate

BUCKET = "alpha-engine-research"
REHEARSAL = "rehearsal-2026-10-15-1"
SM = "arn:aws:states:us-east-1:711398986525:execution:ne-weekly-freshness-pipeline"
REHEARSAL_ARN = f"{SM}:{REHEARSAL}"
PROD_ARN = f"{SM}:a1b2c3d4-0000-1111-2222-333344445555"
NS = namespace_prefix(REHEARSAL)


class _Raw(io.BytesIO):
    """The urllib3-response surface botocore reads: ``stream`` for parsed
    bodies, ``read`` for a streaming GetObject body."""

    def stream(self, **_):
        yield self.getvalue()


class FakeS3:
    """In-memory S3 behind botocore's ``before-send``: records every request's
    final key and answers PUT/GET/HEAD from a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[tuple[str, str]] = []

    def _key(self, url: str) -> str:
        u = urlsplit(url)
        path = unquote(u.path).lstrip("/")
        if u.hostname and u.hostname.startswith(f"{BUCKET}."):
            return path
        return path.split("/", 1)[1] if "/" in path else ""

    def __call__(self, request, **_):
        key = self._key(request.url)
        self.requests.append((request.method, key))
        if request.method == "PUT":
            body = request.body
            if hasattr(body, "read"):
                body = body.read()
            self.objects[key] = body or b""
            return AWSResponse(request.url, 200, {"ETag": '"0"'}, _Raw(b""))
        if key in self.objects:
            data = self.objects[key] if request.method == "GET" else b""
            return AWSResponse(request.url, 200, {"Content-Length": str(len(data))}, _Raw(data))
        err = b"<Error><Code>NoSuchKey</Code><Message>x</Message></Error>"
        return AWSResponse(request.url, 404, {}, _Raw(err if request.method == "GET" else b""))

    def puts(self) -> set[str]:
        return {k for m, k in self.requests if m == "PUT"}


@pytest.fixture(autouse=True)
def fake_s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("RUN_TOKEN", raising=False)
    monkeypatch.delenv("AE_PIPELINE_ROLE", raising=False)
    # Plain request bodies (no aws-chunked checksum trailer) for the fake.
    monkeypatch.setenv("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    boto3.setup_default_session()
    fake = FakeS3()
    boto3.DEFAULT_SESSION._session.register("before-send.s3", fake)
    yield fake
    boto3.DEFAULT_SESSION = None


def _payload() -> dict:
    return {
        "run_id": "2610160202",
        "calendar_date": "2026-10-16",
        "trading_day": "2026-10-15",
        "schema_version": 1,
        "hmm": {"argmax": "neutral"},
        "composite": {"intensity_z": 0.1},
        "bocpd": {"change_signal": False},
    }


@pytest.mark.parametrize(
    "event,expected",
    [
        ({"action": "produce", "execution_arn": REHEARSAL_ARN}, REHEARSAL),
        ({"action": "produce", "execution_arn": PROD_ARN}, None),
        ({"action": "produce"}, None),
        ({"action": "produce", "execution_arn": f"{SM}:watch-rerun-2026-10-17-1"}, None),
    ],
)
def test_marker_comes_from_execution_arn(event, expected):
    assert rehearsal_name(event, {}) == expected


def test_rehearsal_never_writes_regime_latest(fake_s3):
    with isolate({"execution_arn": REHEARSAL_ARN}, stage="RegimeSubstrate") as ctx:
        keys = write_regime_substrate(_payload(), s3_client=boto3.client("s3"), bucket=BUCKET, prefix="regime")
    assert fake_s3.puts() == {f"{NS}regime/2610160202.json", f"{NS}regime/latest.json"}
    assert not any(k.startswith("regime/") for k in fake_s3.objects)
    # The function reports the logical keys; the bytes went to the namespace.
    assert keys["latest_key"] == "regime/latest.json"
    assert ctx.summary()["writes_redirected"] == 2


def test_production_writes_regime_latest_unchanged(fake_s3):
    with isolate({"execution_arn": PROD_ARN}, stage="RegimeSubstrate") as ctx:
        assert ctx is None
        write_regime_substrate(_payload(), s3_client=boto3.client("s3"), bucket=BUCKET, prefix="regime")
    assert fake_s3.puts() == {"regime/2610160202.json", "regime/latest.json"}


def test_rehearsal_reads_its_own_pointer_and_live_inputs(fake_s3):
    fake_s3.objects["regime/latest.json"] = b"PRODUCTION"
    fake_s3.objects["price_cache/SPY.parquet"] = b"LIVE-INPUT"
    with isolate({"execution_arn": REHEARSAL_ARN}, stage="RegimeRetrospectiveEval"):
        c = boto3.client("s3")
        assert c.get_object(Bucket=BUCKET, Key="regime/latest.json")["Body"].read() == b"PRODUCTION"
        c.put_object(Bucket=BUCKET, Key="regime/latest.json", Body=b"REHEARSAL")
        assert c.get_object(Bucket=BUCKET, Key="regime/latest.json")["Body"].read() == b"REHEARSAL"
        assert c.get_object(Bucket=BUCKET, Key="price_cache/SPY.parquet")["Body"].read() == b"LIVE-INPUT"
    assert fake_s3.objects["regime/latest.json"] == b"PRODUCTION"


def test_later_stage_reads_what_regime_substrate_wrote(fake_s3, monkeypatch):
    """RegimeRetrospectiveEval runs after RegimeSubstrate in the same
    execution; the namespace listing at entry makes the earlier write visible."""
    fake_s3.objects[f"{NS}regime/latest.json"] = b"FROM-SUBSTRATE"
    fake_s3.objects["regime/latest.json"] = b"PRODUCTION"

    def prime(self, client, bucket):
        self._overlay.add((bucket, "regime/latest.json"))
        return 1

    monkeypatch.setattr(rehearsal.RehearsalContext, "prime", prime)
    with isolate({"execution_arn": REHEARSAL_ARN}, stage="RegimeRetrospectiveEval"):
        body = boto3.client("s3").get_object(Bucket=BUCKET, Key="regime/latest.json")["Body"].read()
    assert body == b"FROM-SUBSTRATE"


def test_cached_client_is_inert_after_the_rehearsal(fake_s3):
    with isolate({"execution_arn": REHEARSAL_ARN}, stage="RegimeSubstrate"):
        cached = boto3.client("s3")
    cached.put_object(Bucket=BUCKET, Key="regime/latest.json", Body=b"x")
    assert fake_s3.puts() == {"regime/latest.json"}
    assert rehearsal.active() is None


@pytest.mark.parametrize(
    "mod,stage",
    [(regime_handler, "RegimeSubstrate"), (retro_handler, "RegimeRetrospectiveEval")],
)
def test_handlers_isolate_before_creating_their_client(fake_s3, monkeypatch, mod, stage):
    def fake_handle(event, context):
        boto3.client("s3").put_object(Bucket=BUCKET, Key="regime/latest.json", Body=b"x")
        return {"statusCode": 200, "action": "produce"}

    monkeypatch.setattr(mod, "_handle", fake_handle)

    out = mod.lambda_handler({"action": "produce", "execution_arn": REHEARSAL_ARN}, None)
    assert out["rehearsal"]["execution_name"] == REHEARSAL
    assert out["rehearsal"]["stage"] == stage
    assert fake_s3.puts() == {f"{NS}regime/latest.json"}

    out = mod.lambda_handler({"action": "produce", "execution_arn": PROD_ARN}, None)
    assert "rehearsal" not in out
    assert fake_s3.puts() == {f"{NS}regime/latest.json", "regime/latest.json"}


def test_tolerate_stale_only_inside_a_rehearsal():
    assert tolerate_stale("x", where="y") is False
    with isolate({"execution_arn": REHEARSAL_ARN}, stage="RegimeSubstrate") as ctx:
        assert tolerate_stale("macro history 6 days old", where="features") is True
    assert ctx.summary()["degradations"] == [{"where": "features", "refusal": "macro history 6 days old"}]
