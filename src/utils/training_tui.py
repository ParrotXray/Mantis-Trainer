import math
from datetime import timedelta
from time import time
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Footer, Header, ProgressBar, RichLog, Sparkline, Static

# (key, label) for the stat cards across the top of the dashboard.
_STAT_CARDS = [
    ("epoch", "Epoch"),
    ("train_loss", "Train Loss"),
    ("val_loss", "Val Loss"),
    ("best_val", "Best Val Loss"),
    ("val_mae", "Val MAE"),
    ("lr", "Learning Rate"),
    ("elapsed", "Elapsed"),
]


def _fmt(value: Optional[float], spec: str = ".6f") -> str:
    if value is None or math.isnan(value):
        return "—"
    return format(value, spec)


class TrainingDashboard(App):
    """Live training dashboard. Runs on the main thread while the Lightning
    Trainer.fit() call runs on a background thread; all UI mutation from
    that background thread must go through App.call_from_thread()."""

    CSS = """
    Screen { background: $background; }
    #body { padding: 0 1; }
    .panel {
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    #stats {
        grid-size: 7;
        grid-gutter: 0 1;
        height: 5;
        margin-top: 1;
    }
    .stat {
        height: 5;
        padding: 0 1;
        border: round $panel-lighten-2;
        border-title-color: $text-muted;
        background: $surface;
        content-align: center middle;
        text-style: bold;
    }
    #stat-epoch { color: $accent; }
    #stat-train_loss { color: $primary; }
    #stat-val_loss { color: $warning; }
    #stat-best_val { color: $success; }
    #progress-row { height: 5; margin-top: 1; }
    #progress-row .panel { width: 1fr; height: 5; padding: 1 2; }
    #progress-row ProgressBar { width: 100%; }
    #progress-row Bar { width: 1fr; }
    #epoch-bar Bar > .bar--bar, #epoch-bar Bar > .bar--complete { color: $accent; }
    #batch-bar Bar > .bar--bar, #batch-bar Bar > .bar--complete { color: $primary; }
    #chart-row { height: 9; margin-top: 1; }
    #chart-row .panel { width: 1fr; height: 9; }
    #chart-row .panel:first-child, #progress-row .panel:first-child { margin-right: 1; }
    Sparkline { height: 1fr; margin: 1 0; }
    #train-sparkline > .sparkline--max-color { color: $primary; }
    #train-sparkline > .sparkline--min-color { color: $primary 40%; }
    #val-sparkline > .sparkline--max-color { color: $warning; }
    #val-sparkline > .sparkline--min-color { color: $warning 40%; }
    #log-panel { height: 1fr; min-height: 6; margin: 1 0; }
    #log { background: $surface; border: none; scrollbar-size-vertical: 1; }
    """

    BINDINGS = [
        Binding("q", "detach", "Detach (training keeps running in background)"),
    ]

    def __init__(
        self, max_epochs: int, title: str = "Mantis Trainer — LSTM Autoencoder"
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.dashboard_title = title
        self.start_time = time()
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.best_val_loss: Optional[float] = None
        self.error: Optional[BaseException] = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="body"):
            with Grid(id="stats"):
                for key, label in _STAT_CARDS:
                    card = Static("—", id=f"stat-{key}", classes="stat")
                    card.border_title = label
                    yield card
            with Horizontal(id="progress-row"):
                with Vertical(classes="panel", id="epoch-panel") as panel:
                    panel.border_title = "Epochs"
                    yield ProgressBar(
                        total=self.max_epochs, id="epoch-bar", show_eta=False
                    )
                with Vertical(classes="panel", id="batch-panel") as panel:
                    panel.border_title = "Batches"
                    yield ProgressBar(total=100, id="batch-bar", show_eta=False)
            with Horizontal(id="chart-row"):
                with Vertical(classes="panel", id="train-chart") as panel:
                    panel.border_title = "Train Loss"
                    yield Sparkline([], id="train-sparkline")
                with Vertical(classes="panel", id="val-chart") as panel:
                    panel.border_title = "Val Loss"
                    yield Sparkline([], id="val-sparkline")
            with Vertical(classes="panel", id="log-panel") as panel:
                panel.border_title = "Log"
                yield RichLog(id="log", wrap=False, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.dashboard_title
        self.sub_title = "● running"
        self.set_interval(1.0, self._tick_elapsed)

    def _tick_elapsed(self) -> None:
        self._set_stat("elapsed", str(timedelta(seconds=int(time() - self.start_time))))

    def action_detach(self) -> None:
        self.query_one("#log", RichLog).write(
            "[bold yellow]Detaching from dashboard — training keeps running in the background.[/]"
        )
        self.exit()

    def _set_stat(self, key: str, value: str) -> None:
        self.query_one(f"#stat-{key}", Static).update(value)

    def _log(self, message: str) -> None:
        stamp = str(timedelta(seconds=int(time() - self.start_time)))
        self.query_one("#log", RichLog).write(f"[dim]{stamp:>8}[/]  {message}")

    def start_epoch(self, epoch: int, total_batches: int, lr: float) -> None:
        self.query_one("#epoch-bar", ProgressBar).update(progress=epoch)
        self.query_one("#batch-bar", ProgressBar).update(
            total=total_batches, progress=0
        )
        self.query_one("#batch-panel").border_subtitle = f"0/{total_batches}"
        self.query_one("#epoch-panel").border_subtitle = f"{epoch}/{self.max_epochs}"
        self._set_stat("epoch", f"{epoch}/{self.max_epochs}")
        self._set_stat("lr", f"{lr:.2e}")
        self._log(
            f"[bold cyan]▶ Epoch {epoch}/{self.max_epochs}[/] start — "
            f"{total_batches} batches, lr={lr:.2e}"
        )

    def update_batch(self, batch_idx: int, avg_loss: float, lr: float) -> None:
        bar = self.query_one("#batch-bar", ProgressBar)
        bar.update(progress=batch_idx)
        total = int(bar.total) if bar.total is not None else "?"
        self.query_one("#batch-panel").border_subtitle = f"{batch_idx}/{total}"
        self._set_stat("train_loss", _fmt(avg_loss))
        self._set_stat("lr", f"{lr:.2e}")

    def _update_chart(
        self, chart_id: str, panel_id: str, history: List[float], value: float
    ) -> None:
        history.append(value)
        self.query_one(chart_id, Sparkline).data = list(history)
        self.query_one(panel_id).border_subtitle = (
            f"last {_fmt(value)} · min {_fmt(min(history))}"
        )

    def end_epoch(self, epoch: int, metrics: Dict[str, float], elapsed: float) -> None:
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_mae = metrics.get("val_mae")

        if train_loss is not None and not math.isnan(train_loss):
            self._update_chart(
                "#train-sparkline", "#train-chart", self.train_loss_history, train_loss
            )

        improved = False
        if val_loss is not None and not math.isnan(val_loss):
            self._update_chart(
                "#val-sparkline", "#val-chart", self.val_loss_history, val_loss
            )
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improved = True
                self._set_stat("best_val", f"{_fmt(val_loss)} @{epoch}")

        self.query_one("#epoch-bar", ProgressBar).update(progress=epoch + 1)
        self.query_one("#epoch-panel").border_subtitle = (
            f"{epoch + 1}/{self.max_epochs}"
        )
        self._set_stat("train_loss", _fmt(train_loss))
        self._set_stat("val_loss", _fmt(val_loss))
        self._set_stat("val_mae", _fmt(val_mae))
        self._tick_elapsed()

        marker = " [bold green]★ new best[/]" if improved else ""
        self._log(
            f"[bold green]✔ Epoch {epoch}/{self.max_epochs} done[/] "
            f"[dim]({timedelta(seconds=int(elapsed))})[/]  "
            f"train_loss={_fmt(train_loss)}  val_loss={_fmt(val_loss)}  "
            f"val_mae={_fmt(val_mae)}{marker}"
        )

    def log_line(self, message: str) -> None:
        self._log(message)

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.error = error
        if self.is_running:
            if error is not None:
                self.sub_title = "✖ failed"
                self._log(f"[bold red]✖ Training failed: {error}[/]")
            else:
                self.sub_title = "✔ complete"
                self._log("[bold green]✔ Training complete.[/]")
            self.exit()
