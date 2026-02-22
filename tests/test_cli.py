import argparse

import pytest

import manage


def test_update_calls_ingest_before_compute(monkeypatch):
    calls = []

    monkeypatch.setattr(manage, "_stage_ingest_universe", lambda args: calls.append("ingest_universe"))
    monkeypatch.setattr(manage, "_stage_ingest_prices", lambda args: calls.append("ingest_prices"))
    monkeypatch.setattr(manage, "_stage_ingest_fundamentals", lambda args: calls.append("ingest_fundamentals"))
    monkeypatch.setattr(manage, "_stage_compute_factors", lambda args: calls.append("compute_factors"))
    monkeypatch.setattr(manage, "_stage_compute_analytics", lambda args: calls.append("compute_analytics"))

    args = argparse.Namespace(compute_only=False, dry_run=False, portfolio="default")
    manage.cmd_update(args)

    assert calls == [
        "ingest_universe",
        "ingest_prices",
        "ingest_fundamentals",
        "compute_factors",
        "compute_analytics",
    ]


def test_ingest_failure_halts_pipeline(monkeypatch):
    calls = []

    monkeypatch.setattr(manage, "_stage_ingest_universe", lambda args: calls.append("ingest_universe"))

    def _fail_prices(args):
        calls.append("ingest_prices")
        raise RuntimeError("boom")

    monkeypatch.setattr(manage, "_stage_ingest_prices", _fail_prices)
    monkeypatch.setattr(manage, "_stage_ingest_fundamentals", lambda args: calls.append("ingest_fundamentals"))
    monkeypatch.setattr(manage, "_stage_compute_factors", lambda args: calls.append("compute_factors"))
    monkeypatch.setattr(manage, "_stage_compute_analytics", lambda args: calls.append("compute_analytics"))

    args = argparse.Namespace(compute_only=False, dry_run=False, portfolio="default")
    with pytest.raises(SystemExit):
        manage.cmd_update(args)

    assert calls == ["ingest_universe", "ingest_prices"]


def test_compute_only_skips_ingest(monkeypatch):
    calls = []

    monkeypatch.setattr(manage, "_stage_ingest_universe", lambda args: calls.append("ingest_universe"))
    monkeypatch.setattr(manage, "_stage_ingest_prices", lambda args: calls.append("ingest_prices"))
    monkeypatch.setattr(manage, "_stage_ingest_fundamentals", lambda args: calls.append("ingest_fundamentals"))
    monkeypatch.setattr(manage, "_stage_compute_factors", lambda args: calls.append("compute_factors"))
    monkeypatch.setattr(manage, "_stage_compute_analytics", lambda args: calls.append("compute_analytics"))

    args = argparse.Namespace(compute_only=True, dry_run=False, portfolio="default")
    manage.cmd_update(args)

    assert calls == ["compute_analytics"]


def test_dry_run_no_execution(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(manage, "_stage_ingest_universe", lambda args: calls.append("ingest_universe"))
    monkeypatch.setattr(manage, "_stage_ingest_prices", lambda args: calls.append("ingest_prices"))
    monkeypatch.setattr(manage, "_stage_ingest_fundamentals", lambda args: calls.append("ingest_fundamentals"))
    monkeypatch.setattr(manage, "_stage_compute_factors", lambda args: calls.append("compute_factors"))
    monkeypatch.setattr(manage, "_stage_compute_analytics", lambda args: calls.append("compute_analytics"))

    args = argparse.Namespace(compute_only=False, dry_run=True, portfolio="default")
    manage.cmd_update(args)

    assert calls == []
    out = capsys.readouterr().out
    assert "ingest_universe" in out
    assert "compute_analytics" in out
