"""Estimativa e controle de gastos. Nada pago roda sem passar por aqui."""

from __future__ import annotations

from dataclasses import dataclass

from vvf.config import Settings
from vvf.errors import BudgetExceeded, PaidCallsDisabled, UnknownPrice
from vvf.models.manifest import CostEntry, Manifest


@dataclass
class Estimate:
    provider: str
    model: str
    kind: str
    usd: float | None  # None = preço desconhecido (nunca assumimos zero)
    basis: str  # de onde veio o número, para auditoria


class CostGuard:
    """Autoriza (ou recusa) uma chamada paga e registra a decisão no manifesto.

    Ordem das verificações em uma execução real:
      1. chamadas pagas habilitadas? (VVF_ALLOW_PAID_CALLS=true)
      2. preço conhecido? (nunca se assume 0)
      3. gasto desta execução + estimativa <= VVF_MAX_SPEND_USD?
    Em simulação (dry-run) nada é bloqueado: só se registra a estimativa.
    """

    def __init__(self, settings: Settings, manifest: Manifest | None = None):
        self.settings = settings
        self.manifest = manifest
        self.run_spent = 0.0

    def authorize(self, est: Estimate, *, dry_run: bool, params: dict | None = None) -> CostEntry:
        entry = CostEntry(
            provider=est.provider,
            model=est.model,
            kind=est.kind,
            estimated_usd=est.usd,
            dry_run=dry_run,
            params=params or {},
            note=est.basis,
        )
        if not dry_run:
            if not self.settings.vvf_allow_paid_calls:
                raise PaidCallsDisabled(
                    "chamadas pagas desativadas: defina VVF_ALLOW_PAID_CALLS=true "
                    "(e VVF_MAX_SPEND_USD) para executar de verdade"
                )
            if est.usd is None:
                raise UnknownPrice(
                    f"preço de {est.provider}/{est.model} desconhecido ({est.basis}); "
                    "informe um preço máximo com --price-usd ou configure o preço"
                )
            limit = self.settings.vvf_max_spend_usd
            if self.run_spent + est.usd > limit + 1e-9:
                raise BudgetExceeded(
                    f"estimativa US$ {est.usd:.4f} + já gasto US$ {self.run_spent:.4f} "
                    f"excede o teto US$ {limit:.2f} (VVF_MAX_SPEND_USD)"
                )
            self.run_spent += est.usd
        if self.manifest is not None:
            self.manifest.costs.append(entry)
        return entry

    def settle(self, entry: CostEntry, actual_usd: float | None) -> None:
        """Registra o custo real e corrige o acumulado da execução."""
        if actual_usd is None or entry.estimated_usd is None:
            return
        entry.actual_usd = actual_usd
        self.run_spent += actual_usd - entry.estimated_usd
