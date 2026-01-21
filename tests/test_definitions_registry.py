from src.definitions import DEFINITIONS_REGISTRY


def test_definitions_registry_minimum_keys() -> None:
    required = {"twr", "mwr", "factor_tilts", "var_daily", "cvar_daily"}
    missing = required - set(DEFINITIONS_REGISTRY.keys())
    assert not missing
