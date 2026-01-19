import pytest

from src.utils_io import yahoo_ticker


def test_yahoo_ticker_dot_conversion():
    assert yahoo_ticker("BRK.B") == "BRK-B"
    assert yahoo_ticker("BF.B") == "BF-B"
