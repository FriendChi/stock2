from tradition.metrics import align_rf_series_to_returns, compute_return_metrics, save_equity_curve_plot


def test_compute_return_metrics_returns_required_fields():
    metrics = compute_return_metrics([100.0, 102.0, 101.0, 105.0])
    assert set(["cumulative_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown"]).issubset(set(metrics.keys()))


def test_save_equity_curve_plot_creates_file(tmp_path):
    output_path = tmp_path / "equity.png"
    save_equity_curve_plot([100.0, 101.0, 103.0], output_path, title="demo")
    assert output_path.exists()


def test_align_rf_series_to_returns_converts_annual_rate_to_daily():
    returns_index = __import__("pandas").date_range("2024-01-02", periods=2, freq="D")
    rf_series = __import__("pandas").Series([0.0365], index=__import__("pandas").to_datetime(["2024-01-01"]))
    aligned = align_rf_series_to_returns(returns_index=returns_index, rf_series=rf_series)
    assert len(aligned) == 2
    assert float(aligned.iloc[0]) > 0


def test_compute_return_metrics_positive_rf_lowers_sharpe():
    equity_curve = __import__("pandas").Series([100.0, 101.0, 102.0, 103.0], index=__import__("pandas").date_range("2024-01-01", periods=4, freq="D"))
    rf_series = __import__("pandas").Series([0.10], index=__import__("pandas").to_datetime(["2024-01-01"]))
    metrics_without_rf = compute_return_metrics(equity_curve)
    metrics_with_rf = compute_return_metrics(equity_curve, rf_series=rf_series)
    assert metrics_with_rf["sharpe"] < metrics_without_rf["sharpe"]
