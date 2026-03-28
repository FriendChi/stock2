from tradition.metrics import compute_return_metrics, save_equity_curve_plot


def test_compute_return_metrics_returns_required_fields():
    metrics = compute_return_metrics([100.0, 102.0, 101.0, 105.0])
    assert set(["cumulative_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown"]).issubset(set(metrics.keys()))


def test_save_equity_curve_plot_creates_file(tmp_path):
    output_path = tmp_path / "equity.png"
    save_equity_curve_plot([100.0, 101.0, 103.0], output_path, title="demo")
    assert output_path.exists()
