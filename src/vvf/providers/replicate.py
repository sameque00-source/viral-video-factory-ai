"""Adaptador Replicate. O esquema do modelo só é legível com token (consulta gratuita)."""

from __future__ import annotations

from typing import Any

from vvf.costs import Estimate
from vvf.errors import MissingCredential, ProviderError
from vvf.providers.base import JobHandle, JobStatus, TransformRequest
from vvf.providers.prices import price_from_table

VIDEO_PARAMS = ("video", "video_url", "input_video", "video_input")


class ReplicateProvider:
    name = "replicate"

    def __init__(self, api_token: str | None, *, client=None):
        self.api_token = api_token
        self._client = client

    @property
    def client(self):
        if self._client is None:
            if not self.api_token:
                raise MissingCredential("REPLICATE_API_TOKEN não configurado")
            import replicate

            self._client = replicate.Client(api_token=self.api_token)
        return self._client

    def has_credentials(self) -> bool:
        return bool(self.api_token) or self._client is not None

    def input_schema(self, model: str) -> dict | None:
        if not self.has_credentials():
            return None
        try:
            version = self.client.models.get(model).latest_version
            schemas = version.openapi_schema["components"]["schemas"]
            return schemas["Input"]
        except Exception:  # noqa: BLE001 - qualquer falha vira "não verificado"
            return None

    def video_param(self, schema: dict | None) -> str:
        props = (schema or {}).get("properties", {})
        return next((p for p in VIDEO_PARAMS if p in props), "video")

    def validate(self, req: TransformRequest) -> tuple[list[str], list[str]]:
        schema = self.input_schema(req.model)
        if schema is None:
            return [], [
                "esquema do modelo não verificado (requer REPLICATE_API_TOKEN): parâmetros não checados"
            ]
        problems = []
        props = schema.get("properties", {})
        vparam = self.video_param(schema)
        if vparam not in props:
            problems.append(f"não encontrei campo de vídeo de entrada no modelo (aceitos: {sorted(props)})")
        sent = {"prompt": req.prompt, **req.params} if req.prompt else dict(req.params)
        for key in sent:
            if key not in props:
                problems.append(
                    f"parâmetro desconhecido para {req.model}: '{key}' (aceitos: {sorted(props)})"
                )
        for key in schema.get("required", []):
            if key != vparam and key not in sent:
                problems.append(f"parâmetro obrigatório ausente: '{key}'")
        return problems, []

    def estimate(self, req: TransformRequest) -> Estimate:
        usd, note = price_from_table(self.name, req.model, req.duration_s)
        return Estimate(self.name, req.model, "video-transform", usd, note)

    def submit(self, req: TransformRequest) -> JobHandle:
        schema = self.input_schema(req.model)
        with req.input_video.open("rb") as fh:
            uploaded = self.client.files.create(fh)
        url = (uploaded.urls or {}).get("get") if hasattr(uploaded, "urls") else None
        args: dict[str, Any] = {self.video_param(schema): url or uploaded}
        if req.prompt:
            args["prompt"] = req.prompt
        args.update(req.params)
        prediction = self.client.predictions.create(model=req.model, input=args)
        return JobHandle(self.name, req.model, prediction.id)

    def poll(self, handle: JobHandle) -> JobStatus:
        pred = self.client.predictions.get(handle.request_id)
        status = pred.status
        if status == "starting":
            return JobStatus("queued")
        if status == "processing":
            return JobStatus("running")
        if status == "succeeded":
            out = pred.output
            url = out[0] if isinstance(out, list) and out else out
            url = getattr(url, "url", url)
            if not isinstance(url, str) or not url.startswith("http"):
                return JobStatus("failed", f"saída inesperada: {type(out).__name__}")
            return JobStatus("succeeded", output_url=url)
        if status in ("failed", "canceled"):
            return JobStatus("failed", str(pred.error or status))
        raise ProviderError(f"status inesperado do Replicate: {status!r}")
