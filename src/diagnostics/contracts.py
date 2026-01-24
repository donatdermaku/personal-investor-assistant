from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


DiagnosticsVersion = "1.0"


@dataclass
class DiagnosticSignal:
    key: str
    category: str
    severity: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    metrics_used: list[str] = field(default_factory=list)
    as_of: str | None = None
    confidence: float = 0.0
    suggested_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "category": self.category,
            "severity": self.severity,
            "summary": self.summary,
            "evidence": self.evidence,
            "metrics_used": self.metrics_used,
            "as_of": self.as_of,
            "confidence": self.confidence,
        }
        if self.suggested_action:
            payload["suggested_action"] = self.suggested_action
        return payload


def as_of_label(value: date | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
