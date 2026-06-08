import pytest
import requests

from qer.universe.wikipedia import get_universe


def _get_universe_or_skip(date_str: str):
    """Fetch the universe, skipping the test if Wikipedia is unreachable.

    These tests exercise the live scrape, so they depend on network access to
    Wikipedia. A 403/connection error is an environment issue, not a code
    defect, so we skip rather than fail. The reconstruction logic is covered
    hermetically (mocked tables) in tests/integration/test_end_to_end.py.
    """
    try:
        return get_universe(date_str)
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"Wikipedia unreachable ({exc}); skipping live-scrape test.")


def test_universe_length():
    test_dates = ["2010-06-30", "2013-12-31", "2017-06-15", "2020-01-02", "2024-03-01"]

    for d in test_dates:
        universe = _get_universe_or_skip(d)
        print(f"{d}: {len(universe)} tickers (sample: {universe[:5]})")
        # S&P 500 nominally has 500 names; can reach 503-505 because some
        # firms have two share classes (e.g. Alphabet has both GOOG and
        # GOOGL). Allow modest slack but tight enough to catch real bugs.
        assert 460 <= len(universe) <= 540, f"Unexpected universe size {len(universe)} for {d}"


def test_assets_presence():
    # Spot check: BRK-B (Berkshire B) has been in the S&P 500 since Feb 2010.
    # If this fails after the size checks pass, the Yahoo symbol mapping is
    # the most likely culprit (Wikipedia stores it as BRK.B).
    u_2020 = _get_universe_or_skip("2020-01-02")
    assert "BRK-B" in u_2020, (
        "BRK-B missing from 2020 universe - check the Yahoo symbol mapping "
        "and that class-share tickers are being preserved through scraping."
    )
