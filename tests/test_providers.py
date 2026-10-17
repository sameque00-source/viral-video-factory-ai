import json
import os

import httpx
import pytest

from vvf.providers.base import JobHandle, TransformRequest
from vvf.providers.fal import FalProvider
from vvf.providers.prices import price_from_table
from vvf.providers.replicate import ReplicateProvider

SCHEMA_DOC = {
    "components": {
        "schemas": {
            "XInput": {
                "properties": {
                    "video_url": {"type": "string"},
                    "prompt": {"type": "string"},
                    "mode": {"enum": ["flex_1", "flex_2"], "type": "string"},
                },
                "required": ["video_url"],
            }
        }
    }
}


def _http(pricing=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if "openapi" in str(request.url):
            return httpx.Response(status, json=SCHEMA_DOC)
        if "pricing" in str(request.url):
            assert request.headers["authorization"] == "Key k-teste"
            return httpx.Response(200, json={"prices": [pricing]})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _req(tmp_path, **kw):
    f = tmp_path / "in.mp4"
    f.write_bytes(b"x")
    return TransformRequest(f, kw.pop("prompt", "faça X"), "fal-ai/x/modify", kw.pop("params", {}), 10.0)


def test_fal_validates_against_public_schema(tmp_path):
    p = FalProvider(None, http=_http(), cache_dir=tmp_path)
    ok, _ = p.validate(_req(tmp_path, params={"mode": "flex_2"}))
    assert ok == []
    problems, _ = p.validate(_req(tmp_path, params={"mode": "inexistente", "foo": 1}))
    assert any("mode" in x for x in problems) and any("foo" in x for x in problems)


def test_fal_schema_is_cached_on_disk(tmp_path):
    p = FalProvider(None, http=_http(), cache_dir=tmp_path)
    p.input_schema("fal-ai/x/modify")
    offline = FalProvider(None, http=_http(status=500), cache_dir=tmp_path)
    assert offline.input_schema("fal-ai/x/modify") is not None  # veio do disco


def test_fal_unreachable_schema_warns_instead_of_failing(tmp_path):
    p = FalProvider(None, http=_http(status=500))
    problems, warnings = p.validate(_req(tmp_path))
    assert problems == [] and warnings


@pytest.mark.parametrize(
    "unit,price,expected",
    [("video", 0.5, 0.5), ("second", 0.1, 1.0), ("megapixel", 0.2, None)],
)
def test_fal_price_from_official_api(tmp_path, unit, price, expected):
    pricing = {"endpoint_id": "fal-ai/x/modify", "unit_price": price, "unit": unit, "currency": "USD"}
    p = FalProvider("k-teste", http=_http(pricing))
    est = p.estimate(_req(tmp_path))
    assert est.usd == expected


def test_fal_price_unknown_without_key_or_table(tmp_path):
    est = FalProvider(None, http=_http()).estimate(_req(tmp_path))
    assert est.usd is None and "FAL_KEY" in est.basis


def test_user_price_table_takes_precedence(tmp_path, monkeypatch):
    (tmp_path / "provider_prices.json").write_text(
        json.dumps(
            {
                "prices": {
                    "fal:fal-ai/x/modify": {"usd_per_second": 0.2, "verified_at": "hoje", "source": "site"}
                }
            }
        )
    )
    monkeypatch.setenv("VVF_CONFIG_DIR", str(tmp_path))
    assert price_from_table("fal", "fal-ai/x/modify", 5.0)[0] == 1.0
    est = FalProvider(None, http=_http()).estimate(_req(tmp_path))
    assert est.usd == 2.0 and "hoje" in est.basis


class FakeFalSDK:
    def __init__(self):
        self.args = None

    def upload_file(self, path):
        return "https://cdn.exemplo/in.mp4"

    def submit(self, model, arguments):
        self.args = arguments
        return type("H", (), {"request_id": "r1"})()

    def status(self, model, request_id):
        import fal_client

        return fal_client.Completed(logs=None, metrics={})

    def result(self, model, request_id):
        return {"video": {"url": "https://cdn.exemplo/out.mp4"}}


def test_fal_submit_builds_arguments_from_schema_and_poll_reads_result(tmp_path):
    sdk = FakeFalSDK()
    p = FalProvider(None, http=_http(), sdk=sdk, cache_dir=tmp_path)
    handle = p.submit(_req(tmp_path, params={"mode": "flex_1"}))
    assert handle.request_id == "r1"
    assert sdk.args == {"video_url": "https://cdn.exemplo/in.mp4", "prompt": "faça X", "mode": "flex_1"}
    status = p.poll(handle)
    assert status.state == "succeeded" and status.output_url.endswith("out.mp4")


class FakeReplicate:
    def __init__(self, status="succeeded", output="https://replicate.delivery/x.mp4"):
        schema = {"properties": {"video": {}, "prompt": {}}, "required": ["video"]}
        version = type("V", (), {"openapi_schema": {"components": {"schemas": {"Input": schema}}}})()
        self.models = type("M", (), {"get": lambda _s, _m: type("Mo", (), {"latest_version": version})()})()
        self.files = type(
            "F", (), {"create": lambda _s, fh: type("Fi", (), {"urls": {"get": "https://r/f"}})()}
        )()
        pred = type("P", (), {"id": "p1", "status": status, "output": output, "error": "ruim"})()
        self.created = None

        def create(model, input):
            self.created = input
            return pred

        self.predictions = type(
            "Pr", (), {"create": staticmethod(create), "get": staticmethod(lambda _i: pred)}
        )()


def test_replicate_validates_submits_and_polls(tmp_path):
    client = FakeReplicate()
    p = ReplicateProvider(None, client=client)
    req = _req(tmp_path)
    req.model = "luma/modify-video"
    assert p.validate(req) == ([], [])
    bad = _req(tmp_path, params={"zzz": 1})
    problems, _ = p.validate(bad)
    assert any("zzz" in x for x in problems)
    handle = p.submit(req)
    assert client.created == {"video": "https://r/f", "prompt": "faça X"}
    assert p.poll(handle).output_url.startswith("https://replicate.delivery")
    failed = ReplicateProvider(None, client=FakeReplicate(status="failed"))
    assert failed.poll(JobHandle("replicate", "m", "p1")).state == "failed"


def test_replicate_without_token_warns_schema_not_verified(tmp_path):
    problems, warnings = ReplicateProvider(None).validate(_req(tmp_path))
    assert problems == [] and "não verificado" in warnings[0]
    assert ReplicateProvider(None).estimate(_req(tmp_path)).usd is None


@pytest.mark.network
def test_real_fal_public_schema_for_catalog_models(tmp_path):
    from vvf.providers.catalog import CATALOG

    p = FalProvider(None, cache_dir=tmp_path)
    for info in (m for m in CATALOG if m.provider == "fal"):
        schema = p.input_schema(info.model)
        assert schema and p.video_param(schema) in schema["properties"], info.model
    assert os.environ.get("FAL_KEY") is None  # este teste nunca usa chave
