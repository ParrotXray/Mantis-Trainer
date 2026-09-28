"""Post-training plot viewer.

Import this module lazily, right before running the app: textual-image
probes the terminal for Sixel / Kitty graphics support at import time, and
that probe must happen on a real TTY before Textual takes over the screen.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Label, ListItem, ListView, Static
from textual_image.widget import Image

# Filename prefix (before "-<datestamp>.png") -> sidebar label.
_PLOT_NAMES = {
    "deep_ae_analysis": "Score analysis",
    "deep_ae_roc_pr": "ROC / PR curves",
    "deep_ae_cdf": "Score CDF",
    "deep_ae_tsne": "Latent t-SNE",
}
_STAMP_RE = re.compile(r"^(?P<prefix>.+)-(?P<stamp>\d{8}_\d{6})\.png$")


def latest_plot_set(plot_dir: Path = Path("plots")) -> List[Path]:
    """The plots from the most recent training run in plot_dir, in the
    order _PLOT_NAMES lists them."""
    runs: Dict[str, Dict[str, Path]] = {}
    for path in plot_dir.glob("*.png"):
        match = _STAMP_RE.match(path.name)
        if match and match["prefix"] in _PLOT_NAMES:
            runs.setdefault(match["stamp"], {})[match["prefix"]] = path
    if not runs:
        return []
    latest = runs[max(runs)]
    return [latest[prefix] for prefix in _PLOT_NAMES if prefix in latest]


def _label_for(path: Path) -> str:
    match = _STAMP_RE.match(path.name)
    return _PLOT_NAMES.get(match["prefix"], path.stem) if match else path.stem


class ResultsViewer(App):
    """Browse the result plots of a training run: pick one on the left,
    see it on the right, rendered with the best graphics protocol the
    terminal supports (Kitty / Sixel, else Unicode half-blocks)."""

    CSS = """
    Screen { background: $background; }
    #summary {
        height: auto;
        margin: 1 1 0 1;
        padding: 0 1;
        border: round $success;
        border-title-color: $success;
        border-title-style: bold;
        background: $surface;
    }
    #main { margin: 1; }
    #sidebar {
        width: 26;
        margin-right: 1;
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        background: $surface;
    }
    #sidebar:focus-within { border: round $accent; border-title-color: $accent; }
    #plot-list { background: transparent; }
    #plot-list > ListItem { padding: 0 1; background: transparent; }
    #plot-list > ListItem.-highlight { background: $accent 30%; }
    #viewer {
        width: 1fr;
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        background: $surface;
        align: center middle;
    }
    #plot-image { width: auto; height: 1fr; }
    #empty { width: 100%; height: 100%; content-align: center middle; color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Close"),
        Binding("escape", "quit", "Close", show=False),
    ]

    def __init__(self, plots: List[Path], summary: Optional[str] = None) -> None:
        super().__init__()
        self.plots = plots
        self.summary = summary

    def compose(self) -> ComposeResult:
        yield Header()
        if self.summary:
            summary = Static(self.summary, id="summary")
            summary.border_title = "Summary"
            yield summary
        with Horizontal(id="main"):
            with Vertical(id="sidebar") as sidebar:
                sidebar.border_title = "Plots"
                yield ListView(
                    *[ListItem(Label(_label_for(p))) for p in self.plots],
                    id="plot-list",
                )
            with Vertical(id="viewer"):
                if self.plots:
                    yield Image(self.plots[0], id="plot-image")
                else:
                    yield Static("No plots found in ./plots", id="empty")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mantis Trainer"
        self.sub_title = "Training results"
        if self.plots:
            self._show(0)
        self.query_one("#plot-list", ListView).focus()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.index is not None:
            self._show(event.list_view.index)

    def _show(self, index: int) -> None:
        path = self.plots[index]
        self.query_one("#plot-image", Image).image = path
        viewer = self.query_one("#viewer")
        viewer.border_title = _label_for(path)
        viewer.border_subtitle = str(path)
