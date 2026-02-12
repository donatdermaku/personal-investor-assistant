from __future__ import annotations

import pytest

from src.analytics.metrics_registry import (
    assert_metric_artifact_aliases_registered,
    assert_metrics_registered,
    assert_run_metric_payload_keys_registered,
    filter_registered_metrics,
    get_exposed_definitions,
)


def test_exposed_definitions_only_registered_metrics() -> None:
    payload = get_exposed_definitions()
    assert "twr" in payload
    assert "hhi_concentration" in payload
    assert "var_budget_comparison" in payload


def test_assert_metrics_registered_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError):
        assert_metrics_registered(["unknown_metric"])


def test_filter_registered_metrics_drops_unknown_keys() -> None:
    filtered = filter_registered_metrics({"twr": 0.1, "unknown_metric": 42})
    assert filtered == {"twr": 0.1}


def test_assert_run_metric_payload_keys_registered_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        assert_run_metric_payload_keys_registered(["risk", "rogue_metric_blob"])


def test_assert_metric_artifact_aliases_registered_rejects_unknown_alias() -> None:
    with pytest.raises(ValueError):
        assert_metric_artifact_aliases_registered(["benchmark-comparison", "rogue-artifact"])
