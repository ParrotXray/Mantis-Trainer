"""Post-training plot viewer.

Import this module lazily, right before running the app: textual-image
probes the terminal for Sixel / Kitty graphics support at import time, and
that probe must happen on a real TTY before Textual takes over the screen.
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    ContentSwitcher,
    DataTable,
    Footer,
    Header,
    Label,
    ListItem,
    ListView,
    Static,
)
from textual_image.widget import Image

# Filename prefix (before "-<datestamp>.png") -> sidebar label.
_PLOT_NAMES = {
    "deep_ae_analysis": "Score analysis",
    "deep_ae_roc_pr": "ROC / PR curves",
    "deep_ae_cdf": "Score CDF",
    "deep_ae_tsne": "Latent t-SNE",
}
# Written by DeepAutoencoder.save_results() (not imported from there: that
# would pull torch into the viewer).
THRESHOLDS_CSV = Path("outputs") / "deep_ae_thresholds.csv"
PER_CLASS_CSV = Path("outputs") / "deep_ae_per_class_tpr.csv"
_TABLE_ITEM = "Thresholds"

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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pct(value: str) -> str:
    return f"{float(value) * 100:.2f}%"


def _label_for(path: Path) -> str:
    match = _STAMP_RE.match(path.name)
    return _PLOT_NAMES.get(match["prefix"], path.stem) if match else path.stem


class ResultsViewer(App):
    """Browse a training run's results: the threshold table (each val
    threshold applied to the test set, with per-attack-type TPR for the
    highlighted row) and the result plots, rendered with the best graphics
    protocol the terminal supports (Kitty / Sixel, else Unicode half-blocks)."""

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
    #switcher, #image-view, #table-view { width: 100%; height: 100%; }
    #image-view { align: center middle; }
    #threshold-table { height: auto; max-height: 60%; }
    #per-class-title { margin-top: 1; color: $text-muted; text-style: bold; }
    #per-class-table { height: 1fr; }
    .hint { color: $text-muted; }
    #empty { width: 100%; height: 100%; content-align: center middle; color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Close"),
        Binding("escape", "quit", "Close", show=False),
    ]

    def __init__(
        self,
        plots: List[Path],
        summary: Optional[str] = None,
        thresholds_csv: Path = THRESHOLDS_CSV,
        per_class_csv: Path = PER_CLASS_CSV,
    ) -> None:
        super().__init__()
        self.plots = plots
        self.summary = summary
        self.threshold_rows = _read_csv(thresholds_csv)
        self.per_class_rows = _read_csv(per_class_csv)
        # Sidebar entries: the threshold table first (if any), then plots.
        self.entries: List[Optional[Path]] = (
            [None] if self.threshold_rows else []
        ) + list(plots)

    def compose(self) -> ComposeResult:
        yield Header()
        if self.summary:
            summary = Static(self.summary, id="summary")
            summary.border_title = "Summary"
            yield summary
        with Horizontal(id="main"):
            with Vertical(id="sidebar") as sidebar:
                sidebar.border_title = "Results"
                yield ListView(
                    *[
                        ListItem(Label(_label_for(p) if p else _TABLE_ITEM))
                        for p in self.entries
                    ],
                    id="plot-list",
                )
            with Vertical(id="viewer"):
                if not self.entries:
                    yield Static("No plots or tables found", id="empty")
                else:
                    with ContentSwitcher(id="switcher"):
                        with Vertical(id="table-view"):
                            yield Static(
                                "Each val threshold applied to the test set — "
                                "★ best F1, ◆ best Youden. Precision / F1 depend on "
                                "the test set's attack ratio.",
                                classes="hint",
                            )
                            yield DataTable(id="threshold-table", cursor_type="row")
                            yield Static("", id="per-class-title")
                            yield DataTable(id="per-class-table", cursor_type="none")
                        with Vertical(id="image-view"):
                            yield Image(
                                self.plots[0] if self.plots else None, id="plot-image"
                            )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mantis Trainer"
        self.sub_title = "Training results"
        if self.threshold_rows:
            self._fill_threshold_table()
        if self.entries:
            self._show(0)
        self.query_one("#plot-list", ListView).focus()

    def _fill_threshold_table(self) -> None:
        table = self.query_one("#threshold-table", DataTable)
        table.add_columns(
            "",
            "Name",
            "Threshold",
            "Val FPR",
            "FPR",
            "TPR",
            "Precision",
            "F1",
            "Youden",
        )
        best_f1 = max(self.threshold_rows, key=lambda r: float(r["f1"]))
        best_youden = max(self.threshold_rows, key=lambda r: float(r["youden"]))
        for row in self.threshold_rows:
            mark = ("★" if row is best_f1 else "") + ("◆" if row is best_youden else "")
            table.add_row(
                mark,
                row["name"],
                f"{float(row['threshold']):.6f}",
                _pct(row["val_fpr"]),
                _pct(row["test_fpr"]),
                _pct(row["test_tpr"]),
                f"{float(row['precision']):.4f}",
                f"{float(row['f1']):.4f}",
                f"{float(row['youden']):.4f}",
                key=row["name"],
            )
        table.move_cursor(row=self.threshold_rows.index(best_f1))

        per_class = self.query_one("#per-class-table", DataTable)
        per_class.add_columns("Attack type", "TPR", "Samples")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id != "threshold-table" or event.row_key.value is None:
            return
        name = event.row_key.value
        rows = [r for r in self.per_class_rows if r["threshold_name"] == name]
        per_class = self.query_one("#per-class-table", DataTable)
        per_class.clear()
        for r in sorted(rows, key=lambda r: float(r["tpr"])):
            per_class.add_row(
                r["attack_type"], _pct(r["tpr"]), f"{int(r['samples']):,}"
            )
        self.query_one("#per-class-title", Static).update(
            f"Per-attack-type TPR @ {name}  (lowest first)"
        )

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.index is not None:
            self._show(event.list_view.index)

    def _show(self, index: int) -> None:
        entry = self.entries[index]
        viewer = self.query_one("#viewer")
        switcher = self.query_one("#switcher", ContentSwitcher)
        if entry is None:
            switcher.current = "table-view"
            viewer.border_title = "Threshold analysis"
            viewer.border_subtitle = str(THRESHOLDS_CSV)
            return
        switcher.current = "image-view"
        self.query_one("#plot-image", Image).image = entry
        viewer.border_title = _label_for(entry)
        viewer.border_subtitle = str(entry)
