from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("kitcat")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from numpy.typing import NDArray

from wtgen.dsp.mipmap import build_mipmap
from wtgen.dsp.process import align_to_zero_crossing
from wtgen.dsp.waves import (
    generate_polyblep_sawtooth_wavetable,
    generate_sawtooth_wavetable,
)
from wtgen.types import WavetableTables


@dataclass
class PlotSeries:
    """Represents a single line/curve within a plot."""

    label: str
    x: NDArray[np.floating]
    y: NDArray[np.floating]
    style: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, max_points: int | None = None) -> dict[str, Any]:
        """Return a lightweight, LLM-friendly summary of the series."""
        x_values = self.x.tolist()
        y_values = self.y.tolist()
        if max_points is not None and len(x_values) > max_points:
            step = max(len(x_values) // max_points, 1)
            x_values = x_values[::step][:max_points]
            y_values = y_values[::step][:max_points]

        return {
            "label": self.label,
            "points": list(zip(x_values, y_values, strict=False)),
            "style": self.style,
        }


@dataclass
class PlotPanel:
    """Configuration for a single subplot/panel."""

    title: str
    series: list[PlotSeries]
    xlabel: str = "Samples"
    ylabel: str = "Amplitude"
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    legend: bool = True
    grid: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def preview(self, max_points: int = 8) -> dict[str, Any]:
        return {
            "title": self.title,
            "series": [series.to_dict(max_points=max_points) for series in self.series],
            "metadata": self.metadata,
        }


@dataclass
class PlotFigure:
    """A collection of panels ready for rendering or textual inspection."""

    panels: list[PlotPanel]
    layout: tuple[int, int] | None = None
    figsize: tuple[float, float] = (12.0, 5.0)
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self, max_points: int = 8) -> str:
        lines = []
        if self.title:
            lines.append(f"Figure: {self.title}")
        for idx, panel in enumerate(self.panels):
            lines.append(f" Panel {idx}: {panel.title}")
            for series in panel.series:
                preview = series.to_dict(max_points=max_points)
                first_points = preview["points"][: min(max_points, len(preview["points"]))]
                lines.append(
                    f"  · {series.label}: {len(series.x)} samples, preview={first_points}"
                )
        return "\n".join(lines)

    def to_dict(self, max_points: int = 64) -> dict[str, Any]:
        return {
            "title": self.title,
            "metadata": self.metadata,
            "panels": [panel.preview(max_points=max_points) for panel in self.panels],
        }


def _resolve_layout(panel_count: int, layout: tuple[int, int] | None) -> tuple[int, int]:
    if layout:
        rows, cols = layout
    else:
        rows = panel_count
        cols = 1

    if rows * cols < panel_count:
        rows = panel_count
        cols = 1

    return rows, cols


def render_figure(
    figure: PlotFigure,
    *,
    show: bool = True,
    output_path: Path | None = None,
) -> Figure:
    """Render a figure to matplotlib while remaining test/LLM friendly."""

    rows, cols = _resolve_layout(len(figure.panels), figure.layout)
    fig, axes = plt.subplots(rows, cols, figsize=figure.figsize, squeeze=False)

    for idx, panel in enumerate(figure.panels):
        ax = axes[idx // cols][idx % cols]
        for series in panel.series:
            ax.plot(series.x, series.y, label=series.label, **series.style)

        ax.set_title(panel.title)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.grid(panel.grid, alpha=0.3)

        if panel.xlim:
            ax.set_xlim(*panel.xlim)
        if panel.ylim:
            ax.set_ylim(*panel.ylim)

        if panel.legend and any(series.label for series in panel.series):
            ax.legend()

    if figure.title:
        fig.suptitle(figure.title)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def _waveform_panel(
    wave: NDArray[np.floating],
    table: NDArray[np.floating] | None = None,
    *,
    label: str = "waveform",
    title: str = "Waveform",
) -> PlotPanel:
    if table is None:
        table = np.linspace(0, 2 * np.pi, len(wave), endpoint=False)

    series = PlotSeries(label=label, x=table, y=wave, style={"linewidth": 2})
    metadata = {
        "samples": len(wave),
        "rms": float(np.sqrt(np.mean(wave**2))) if len(wave) else 0.0,
        "peak": float(np.max(np.abs(wave))) if len(wave) else 0.0,
    }

    return PlotPanel(
        title=title,
        xlabel="Radians",
        ylabel="Amplitude",
        series=[series],
        xlim=(float(table[0]), float(table[-1])) if len(table) else None,
        metadata=metadata,
    )


def _mipmap_overlay_panel(
    mipmaps: Sequence[NDArray[np.floating]],
    *,
    max_levels: int | None = None,
    title: str = "Mipmap Levels",
    label_prefix: str = "level",
) -> PlotPanel:
    series: list[PlotSeries] = []
    rms_levels = []

    for idx, level in enumerate(mipmaps):
        if max_levels is not None and idx >= max_levels:
            break
        aligned = align_to_zero_crossing(level)
        x_values = np.linspace(0, 2 * np.pi, len(aligned), endpoint=False)
        series.append(
            PlotSeries(
                label=f"{label_prefix} {idx}",
                x=x_values,
                y=aligned,
                style={"alpha": max(0.3, 1.0 - idx * 0.1)},
            )
        )
        rms_levels.append(float(np.sqrt(np.mean(aligned**2))) if len(aligned) else 0.0)

    metadata = {
        "levels": len(series),
        "rms_levels": rms_levels,
    }

    return PlotPanel(
        title=title,
        xlabel="Radians",
        ylabel="Amplitude",
        series=series,
        metadata=metadata,
    )


def waveform_plot(
    table: NDArray[np.floating],
    wave: NDArray[np.floating],
    *,
    title: str = "Waveform",
    render: bool = True,
    show: bool = True,
    output_path: Path | None = None,
) -> PlotFigure:
    panel = _waveform_panel(wave, table, title=title)
    figure = PlotFigure(panels=[panel], title=title)

    if render:
        render_figure(figure, show=show, output_path=output_path)

    return figure


def compare_sawtooth_methods(
    frequency: float = 2,
    *,
    zoom_window: int = 50,
    render: bool = True,
    show: bool = True,
    output_path: Path | None = None,
) -> PlotFigure:
    _, naive = generate_sawtooth_wavetable(frequency)
    _, bandlimited = generate_polyblep_sawtooth_wavetable(frequency)
    t = np.linspace(0, 2 * np.pi, len(naive))

    full_panel = PlotPanel(
        title=f"Sawtooth Comparison @ {frequency} cycles",
        series=[
            PlotSeries("Naive", t, naive, {"alpha": 0.7}),
            PlotSeries("PolyBLEP", t, bandlimited, {"alpha": 0.9}),
        ],
        xlabel="Radians",
        ylabel="Amplitude",
    )

    discontinuity_indices = [
        i for i in range(1, len(naive)) if naive[i] < naive[i - 1]
    ]

    if discontinuity_indices:
        disc_idx = discontinuity_indices[0]
        start_idx = max(0, disc_idx - zoom_window)
        end_idx = min(len(t), disc_idx + zoom_window)
        zoom_panel = PlotPanel(
            title="Phase Discontinuity Zoom",
            series=[
                PlotSeries(
                    "Naive (zoom)",
                    t[start_idx:end_idx],
                    naive[start_idx:end_idx],
                    {"marker": "o", "linewidth": 2, "markersize": 3},
                ),
                PlotSeries(
                    "PolyBLEP (zoom)",
                    t[start_idx:end_idx],
                    bandlimited[start_idx:end_idx],
                    {"marker": "s", "linewidth": 2, "markersize": 3},
                ),
            ],
            xlabel="Radians",
            ylabel="Amplitude",
        )
        panels = [full_panel, zoom_panel]
    else:
        panels = [full_panel]

    figure = PlotFigure(panels=panels, title="Sawtooth Method Comparison")
    if render:
        render_figure(figure, show=show, output_path=output_path)
    return figure


def plot_mipmaps(
    wave: NDArray[np.floating],
    *,
    num_octaves: int = 3,
    rolloff: str = "raised_cosine",
    max_levels: int | None = None,
    render: bool = True,
    show: bool = True,
    output_path: Path | None = None,
) -> PlotFigure:
    mipmaps = build_mipmap(wave, num_octaves=num_octaves, rolloff_method=rolloff)
    panel = _mipmap_overlay_panel(mipmaps, max_levels=max_levels)
    figure = PlotFigure(panels=[panel], title=f"Mipmap Levels ({rolloff})")

    if render:
        render_figure(figure, show=show, output_path=output_path)

    return figure


def compare_mipmap_methods(
    *,
    num_octaves: int = 3,
    methods: Sequence[str] = ("tukey", "blackman", "raised_cosine", "hann"),
    render: bool = True,
    show: bool = True,
    output_path: Path | None = None,
) -> PlotFigure:
    _, base_wave = generate_sawtooth_wavetable()
    panels = []
    for method in methods:
        mipmaps = build_mipmap(base_wave, num_octaves=num_octaves, rolloff_method=method)
        panels.append(
            _mipmap_overlay_panel(
                mipmaps,
                max_levels=num_octaves + 1,
                title=f"{method.replace('_', ' ').title()} Rolloff",
                label_prefix="level",
            )
        )

    rows = int(np.ceil(len(panels) / 2))
    figure = PlotFigure(panels=panels, layout=(rows, 2), title="Rolloff Comparison")

    if render:
        render_figure(figure, show=show, output_path=output_path)

    return figure


def tables_plot_figure(
    tables: WavetableTables,
    *,
    table_id: str = "base",
    max_levels: int | None = 6,
    render: bool = False,
    show: bool = False,
    output_path: Path | None = None,
) -> PlotFigure:
    """Build a figure directly from CLI wavetable tables."""

    if table_id not in tables:
        raise KeyError(f"Table '{table_id}' not found in wavetable export")

    panel = _mipmap_overlay_panel(
        tables[table_id],
        max_levels=max_levels,
        title=f"Wavetable '{table_id}'",
        label_prefix="mip",
    )
    figure = PlotFigure(
        panels=[panel],
        title=f"CLI Export: {table_id}",
        metadata={"table_id": table_id},
    )

    if render:
        render_figure(figure, show=show, output_path=output_path)

    return figure


__all__ = [
    "PlotFigure",
    "PlotPanel",
    "PlotSeries",
    "compare_mipmap_methods",
    "compare_sawtooth_methods",
    "plot_mipmaps",
    "render_figure",
    "tables_plot_figure",
    "waveform_plot",
]
