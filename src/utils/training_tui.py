from datetime import timedelta
from time import time
from typing import Dict, List, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    ProgressBar,
    RichLog,
    Sparkline,
)

_STAT_ROWS = [
    ("epoch", "Epoch"),
    ("lr", "Learning Rate"),
    ("train_loss", "Train Loss"),
    ("val_loss", "Val Loss"),
    ("val_mae", "Val MAE"),
    ("elapsed", "Elapsed"),
]


class TrainingDashboard(App):
    """Live training dashboard. Runs on the main thread while the Lightning
    Trainer.fit() call runs on a background thread; all UI mutation from
    that background thread must go through App.call_from_thread()."""

    CSS = """
    #progress-row { height: 5; }
    .progress-box { width: 1fr; padding: 0 1; }
    #stats-table { height: 9; }
    #chart-row { height: 8; }
    .chart-box { width: 1fr; padding: 0 1; }
    #log { border: solid $accent; }
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
        self.error: Optional[BaseException] = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            with Horizontal(id="progress-row"):
                with Vertical(classes="progress-box"):
                    yield Label("Epoch")
                    yield ProgressBar(
                        total=self.max_epochs, id="epoch-bar", show_eta=False
                    )
                with Vertical(classes="progress-box"):
                    yield Label("Batch")
                    yield ProgressBar(total=100, id="batch-bar", show_eta=False)
            yield DataTable(id="stats-table")
            with Horizontal(id="chart-row"):
                with Vertical(classes="chart-box"):
                    yield Label("Train Loss")
                    yield Sparkline([], id="train-sparkline")
                with Vertical(classes="chart-box"):
                    yield Label("Val Loss")
                    yield Sparkline([], id="val-sparkline")
            yield RichLog(id="log", wrap=False, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.dashboard_title
        table = self.query_one("#stats-table", DataTable)
        table.add_column("Metric", key="metric")
        table.add_column("Value", key="value")
        table.cursor_type = "none"
        for key, label in _STAT_ROWS:
            table.add_row(label, "—", key=key)

    def action_detach(self) -> None:
        self.query_one("#log", RichLog).write(
            "[bold yellow]Detaching from dashboard — training keeps running in the background.[/]"
        )
        self.exit()

    def _set_stat(self, key: str, value: str) -> None:
        table = self.query_one("#stats-table", DataTable)
        table.update_cell(key, "value", value)

    def start_epoch(self, epoch: int, total_batches: int, lr: float) -> None:
        self.query_one("#epoch-bar", ProgressBar).update(progress=epoch)
        self.query_one("#batch-bar", ProgressBar).update(
            total=total_batches, progress=0
        )
        self._set_stat("epoch", f"{epoch}/{self.max_epochs}")
        self._set_stat("lr", f"{lr:.2e}")
        self.query_one("#log", RichLog).write(
            f"[bold cyan]Epoch {epoch}/{self.max_epochs}[/] start — "
            f"{total_batches} batches, lr={lr:.2e}"
        )

    def update_batch(self, batch_idx: int, avg_loss: float, lr: float) -> None:
        self.query_one("#batch-bar", ProgressBar).update(progress=batch_idx)
        self._set_stat("train_loss", f"{avg_loss:.6f}")
        self._set_stat("lr", f"{lr:.2e}")

    def end_epoch(self, epoch: int, metrics: Dict[str, float], elapsed: float) -> None:
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_mae = metrics.get("val_mae")

        if train_loss is not None:
            self.train_loss_history.append(train_loss)
            self.query_one("#train-sparkline", Sparkline).data = self.train_loss_history
        if val_loss is not None:
            self.val_loss_history.append(val_loss)
            self.query_one("#val-sparkline", Sparkline).data = self.val_loss_history

        self._set_stat(
            "train_loss", f"{train_loss:.6f}" if train_loss is not None else "—"
        )
        self._set_stat("val_loss", f"{val_loss:.6f}" if val_loss is not None else "—")
        self._set_stat("val_mae", f"{val_mae:.6f}" if val_mae is not None else "—")
        self._set_stat("elapsed", str(timedelta(seconds=int(time() - self.start_time))))

        self.query_one("#log", RichLog).write(
            f"[bold green]Epoch {epoch}/{self.max_epochs} done[/] "
            f"({timedelta(seconds=int(elapsed))}) "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_mae={val_mae:.6f}"
        )

    def log_line(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    def finish(self, error: Optional[BaseException] = None) -> None:
        self.error = error
        if self.is_running:
            if error is not None:
                self.query_one("#log", RichLog).write(
                    f"[bold red]Training failed: {error}[/]"
                )
            else:
                self.query_one("#log", RichLog).write(
                    "[bold green]Training complete.[/]"
                )
            self.exit()
