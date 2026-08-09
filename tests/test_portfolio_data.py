import pandas as pd

from backtester.portfolio_data import (
    close_price_panel,
    download_yfinance_ohlcv_many,
    ensure_yfinance_ohlcv_csvs,
    load_ohlcv_universe,
)


def ohlcv(index: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": index,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1,
        }
    )


def test_load_universe_uses_shared_dates_and_builds_close_panel(tmp_path) -> None:
    first = tmp_path / "aaa.csv"
    second = tmp_path / "bbb.csv"
    ohlcv(pd.date_range("2024-01-01", periods=3, freq="D"), [1.0, 2.0, 3.0]).to_csv(first, index=False)
    ohlcv(pd.date_range("2024-01-02", periods=3, freq="D"), [4.0, 5.0, 6.0]).to_csv(second, index=False)

    universe = load_ohlcv_universe({"AAA": first, "BBB": second})
    prices = close_price_panel(universe)

    assert len(universe["AAA"]) == 2
    assert list(prices.index) == list(pd.date_range("2024-01-02", periods=2, freq="D"))
    assert list(prices.columns) == ["AAA", "BBB"]


def test_download_many_writes_one_csv_per_ticker(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_download(**kwargs) -> pd.DataFrame:
        calls.append(kwargs["ticker"])
        data = pd.DataFrame(
            {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]},
            index=pd.DatetimeIndex(["2024-01-02"], name="date"),
        )
        data.to_csv(kwargs["output_path"], index_label="date")
        return data

    monkeypatch.setattr("backtester.portfolio_data.download_yfinance_ohlcv", fake_download)
    result = download_yfinance_ohlcv_many(
        ["AAA", "BBB"], "2024-01-01", "2024-01-03", output_directory=tmp_path
    )

    assert calls == ["AAA", "BBB"]
    assert set(result) == {"AAA", "BBB"}
    assert (tmp_path / "AAA.csv").exists()
    assert (tmp_path / "BBB.csv").exists()


def test_ensure_csvs_downloads_only_missing_tickers(monkeypatch, tmp_path) -> None:
    existing = tmp_path / "AAA.csv"
    existing.write_text("already downloaded", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_download_many(tickers, **kwargs) -> dict[str, pd.DataFrame]:
        calls.append(list(tickers))
        for ticker in tickers:
            (tmp_path / f"{ticker}.csv").write_text("downloaded", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        "backtester.portfolio_data.download_yfinance_ohlcv_many",
        fake_download_many,
    )

    paths = ensure_yfinance_ohlcv_csvs(
        ["AAA", "BBB"],
        start="2024-01-01",
        end="2024-02-01",
        interval="1d",
        output_directory=tmp_path,
    )

    assert calls == [["BBB"]]
    assert paths == {"AAA": existing, "BBB": tmp_path / "BBB.csv"}
