"""Adaptador fal.ai. Valida entradas contra o OpenAPI público do endpoint (grátis, sem chave)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from vvf.costs import Estimate
from vvf.errors import MissingCredential, ProviderError
from vvf.providers.base import JobHandle, JobStatus, TransformRequest
from vvf.providers.prices import price_from_table

OPENAPI_URL = "https://fal.ai/api/openapi/queue/openapi.json"
PRICING_URL = "https://api.fal.ai/v1/models/pricing"
VIDEO_PARAMS = ("video_url", "video")


class FalProvider:
    name = "fal"

    def __init__(self, api_key: str | None, *, http: httpx.Client | None = None, sdk=None,
                 cache_dir: Path | None = None):  # fmt: skip
        self.api_key = api_key
        self.http = http or httpx.Client(timeout=30.0)
        self._sdk = sdk
        self.cache_dir = cache_dir
        self._schemas: dict[str, dict] = {}

    # --- esquema público ---
    def input_schema(self, model: str) -> dict | None:
        if model in self._schemas:
            return self._schemas[model]
        cache = (
            self.cache_dir / "fal_schemas" / (model.replace("/", "__") + ".json") if self.cache_dir else None
        )
        if cache and cache.exists():
            self._schemas[model] = json.loads(cache.read_text())
            return self._schemas[model]
        try:
            resp = self.http.get(OPENAPI_URL, params={"endpoint_id": model})
            resp.raise_for_status()
            schemas = resp.json()["components"]["schemas"]
            key = next(k for k in schemas if k.endswith("Input"))
            schema = schemas[key]
        except (httpx.HTTPError, KeyError, StopIteration, ValueError):
            return None
        self._schemas[model] = schema
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(schema))
        return schema

    def video_param(self, schema: dict | None) -> str:
        props = (schema or {}).get("properties", {})
        return next((p for p in VIDEO_PARAMS if p in props), "video_url")

    def validate(self, req: TransformRequest) -> tuple[list[str], list[str]]:
        problems: list[str] = []
        warnings: list[str] = []
        schema = self.input_schema(req.model)
        if schema is None:
            warnings.append("não foi possível obter o esquema do endpoint: parâmetros não verificados")
            return problems, warnings
        props = schema.get("properties", {})
        vparam = self.video_param(schema)
        if vparam not in props:
            problems.append(f"o endpoint não aceita vídeo de entrada ({VIDEO_PARAMS})")
        sent = {"prompt": req.prompt, **req.params} if req.prompt else dict(req.params)
        for key, value in sent.items():
            if key not in props:
                problems.append(
                    f"parâmetro desconhecido para {req.model}: '{key}' (aceitos: {sorted(props)})"
                )
                continue
            enum = props[key].get("enum")
            if enum and value not in enum:
                problems.append(f"valor inválido para '{key}': {value!r} (aceitos: {enum})")
        for key in schema.get("required", []):
            if key != vparam and key not in sent:
                problems.append(f"parâmetro obrigatório ausente: '{key}'")
        return problems, warnings

    # --- preço ---
    def estimate(self, req: TransformRequest) -> Estimate:
        table, note = price_from_table(self.name, req.model, req.duration_s)
        if table is not None:
            return Estimate(self.name, req.model, "video-transform", table, note)
        if not self.api_key:
            return Estimate(self.name, req.model, "video-transform", None,
                            "preço indisponível: defina FAL_KEY (consulta gratuita) "
                            "ou preencha config/provider_prices.json")  # fmt: skip
        try:
            resp = self.http.get(PRICING_URL, params={"endpoint_id": req.model},
                                 headers={"Authorization": f"Key {self.api_key}"})  # fmt: skip
            resp.raise_for_status()
            price = resp.json()["prices"][0]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            return Estimate(self.name, req.model, "video-transform", None, f"falha ao consultar preço: {exc}")
        unit, unit_price = str(price.get("unit", "")).lower(), float(price["unit_price"])
        if price.get("currency", "USD") != "USD":
            return Estimate(self.name, req.model, "video-transform", None, f"moeda {price['currency']}")
        if unit in ("video", "request", "run"):
            return Estimate(self.name, req.model, "video-transform", round(unit_price, 6),
                            f"API de preços do fal: {unit_price}/{unit}")  # fmt: skip
        if unit in ("second", "seconds", "sec"):
            return Estimate(self.name, req.model, "video-transform", round(unit_price * req.duration_s, 6),
                            f"API de preços do fal: {unit_price}/s × {req.duration_s:.1f}s")  # fmt: skip
        return Estimate(
            self.name, req.model, "video-transform", None, f"unidade de cobrança desconhecida: {unit!r}"
        )

    # --- execução (PAGA) ---
    @property
    def sdk(self):
        if self._sdk is None:
            if not self.api_key:
                raise MissingCredential("FAL_KEY não configurada")
            import fal_client

            self._sdk = fal_client.SyncClient(key=self.api_key)
        return self._sdk

    def has_credentials(self) -> bool:
        return bool(self.api_key) or self._sdk is not None

    def submit(self, req: TransformRequest) -> JobHandle:
        schema = self.input_schema(req.model)
        args: dict[str, Any] = {self.video_param(schema): self.sdk.upload_file(req.input_video)}
        if req.prompt:
            args["prompt"] = req.prompt
        args.update(req.params)
        handle = self.sdk.submit(req.model, arguments=args)
        return JobHandle(self.name, req.model, handle.request_id)

    def poll(self, handle: JobHandle) -> JobStatus:
        import fal_client

        status = self.sdk.status(handle.model, handle.request_id)
        if isinstance(status, fal_client.Queued):
            return JobStatus("queued", f"posição {status.position}")
        if isinstance(status, fal_client.InProgress):
            return JobStatus("running")
        if isinstance(status, fal_client.Completed):
            if getattr(status, "error", None):
                return JobStatus("failed", str(status.error))
            result = self.sdk.result(handle.model, handle.request_id)
            url = (result.get("video") or {}).get("url") if isinstance(result, dict) else None
            if not url:
                return JobStatus("failed", "resposta sem video.url")
            return JobStatus("succeeded", output_url=url)
        raise ProviderError(f"status inesperado do fal: {status!r}")
