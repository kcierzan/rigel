import numpy as np

from wtgen.plotting import (
    PlotFigure,
    PlotPanel,
    PlotSeries,
    compare_sawtooth_methods,
    render_figure,
    tables_plot_figure,
    waveform_plot,
)


def test_waveform_plot_summary_includes_metadata():
    table = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    wave = np.sin(table)

    figure = waveform_plot(table, wave, render=False)

    summary = figure.summary(max_points=3)
    assert "Figure" in summary
    assert "waveform" in summary
    assert "128" in summary  # sample count preview


def test_render_figure_creates_file(tmp_path):
    x = np.linspace(0, 1, 16)
    y = x**2
    panel = PlotPanel(title="test", series=[PlotSeries("curve", x, y)])
    figure = PlotFigure(panels=[panel], title="Test Figure")

    output_path = tmp_path / "plot.png"
    render_figure(figure, show=False, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_tables_plot_figure_uses_table_id():
    tables = {
        "base": [np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False)), np.zeros(64)],
        "alt": [np.cos(np.linspace(0, 2 * np.pi, 64, endpoint=False))],
    }

    figure = tables_plot_figure(tables, table_id="alt", render=False)

    assert figure.metadata["table_id"] == "alt"
    assert figure.panels[0].metadata["levels"] == 1


def test_compare_sawtooth_methods_returns_panels():
    figure = compare_sawtooth_methods(render=False)

    assert len(figure.panels) >= 1
    assert any("Sawtooth" in panel.title for panel in figure.panels)
