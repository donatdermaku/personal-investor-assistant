import pytest
from src.analytics.contracts import evaluate_metric_status, KPI_DEPENDENCIES

def test_metric_sufficient():
    # Scenario: Prices sufficient, Benchmark sufficient
    coverage = {
        "prices": {"status": "sufficient", "reason_codes": ["OK"]},
        "benchmark": {"status": "sufficient", "reason_codes": ["OK"]}
    }
    status, reasons = evaluate_metric_status(coverage, ["prices"])
    assert status == "sufficient"
    
    status, reasons = evaluate_metric_status(coverage, ["prices", "benchmark"])
    assert status == "sufficient"

def test_metric_available_low_coverage():
    # Scenario: Prices sufficient but Benchmark insufficient (but present)
    # Status "insufficient" with reason "BENCHMARK_INSUFFICIENT" (Warning)
    coverage = {
        "prices": {"status": "sufficient", "reason_codes": ["OK"]},
        "benchmark": {"status": "insufficient", "reason_codes": ["BENCHMARK_INSUFFICIENT"]}
    }
    status, reasons = evaluate_metric_status(coverage, ["prices", "benchmark"])
    assert status == "available_low_coverage"
    assert "BENCHMARK_INSUFFICIENT" in reasons

def test_metric_unavailable():
    # Scenario: Benchmark missing entirely
    coverage = {
        "prices": {"status": "sufficient", "reason_codes": ["OK"]},
        "benchmark": {"status": "insufficient", "reason_codes": ["BENCHMARK_MISSING"]}
    }
    status, reasons = evaluate_metric_status(coverage, ["prices", "benchmark"])
    assert status == "unavailable"
    assert "BENCHMARK_MISSING" in reasons

def test_metric_unavailable_no_prices():
    # Scenario: No prices at all
    coverage = {
        "prices": {"status": "insufficient", "reason_codes": ["NO_PRICES"]}
    }
    status, reasons = evaluate_metric_status(coverage, ["prices"])
    assert status == "unavailable"
