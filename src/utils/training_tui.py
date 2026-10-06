import math
from datetime import timedelta
from time import time
from typing import Dict, List, Optional, Tuple

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Grid, Horizontal, Vertical
from textual.widgets import Footer, Header, ProgressBar, RichLog, Static
from textual_plotext import PlotextPlot

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


def _log10(value: float) -> float:
    # Losses are non-negative; clamp so an exact 0 can't break the axis.
    return math.log10(max(value, 1e-12))


def _log_ticks(lo: float, hi: float, n: int = 5) -> Tuple[List[float], List[str]]:
    """Evenly spaced ticks across [lo, hi] in log10 space, labelled with the
    original (un-logged) loss values."""
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    positions = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return positions, [f"{10 ** p:.3g}" for p in positions]


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
    #progress-row .panel:first-child { margin-right: 1; }
    #loss-panel { height: 16; min-height: 10; margin-top: 1; padding: 0; }
    #loss-plot { background: $surface; }
    #log-panel { height: 1fr; min-height: 6; margin: 1 0; }
    #log { background: $surface; border: none; scrollbar-size-vertical: 1; }
    """

    BINDINGS = [
        Binding("q", "detach", "Detach (training keeps running in background)"),
        Binding("l", "toggle_log_scale", "Log/linear loss axis"),
    ]

    def __init__(
        self, max_epochs: int, title: str = "Mantis Trainer — LSTM Autoencoder"
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.dashboard_title = title
        self.start_time = time()
        # (epoch, loss) pairs — train and val can each be missing on an epoch.
        self.train_loss_history: List[Tuple[int, float]] = []
        self.val_loss_history: List[Tuple[int, float]] = []
        self.best_val_epoch: Optional[int] = None
        self.log_scale = True
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
            with Vertical(classes="panel", id="loss-panel") as panel:
                panel.border_title = "Loss"
                yield PlotextPlot(id="loss-plot")
            with Vertical(classes="panel", id="log-panel") as panel:
                panel.border_title = "Log"
                yield RichLog(id="log", wrap=False, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = self.dashboard_title
        self.sub_title = "● running"
        self.set_interval(1.0, self._tick_elapsed)
        self._redraw_loss()

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

    def action_toggle_log_scale(self) -> None:
        self.log_scale = not self.log_scale
        self._redraw_loss()

    def _redraw_loss(self) -> None:
        plot = self.query_one("#loss-plot", PlotextPlot)
        plt = plot.plt
        plt.clear_data()
        plt.xlabel("epoch")
        has_data = bool(self.train_loss_history or self.val_loss_history)
        # Never use plotext's own yscale("log"): its build step writes the
        # log-transformed values back into the stored series, so every
        # re-render (terminal resize, layout change) takes log10 again until
        # it hits a negative number and raises "math domain error". Plot
        # log10 values on a linear axis instead and label the ticks with the
        # real losses.
        plt.yscale("linear")
        to_y = _log10 if self.log_scale else (lambda v: v)
        plotted: List[float] = []
        # No plotext labels: its legend box sits top-left, exactly where the
        # high early-epoch losses are drawn. The legend lives in the panel
        # title instead.
        for history, color in (
            (self.train_loss_history, "cyan"),
            (self.val_loss_history, "orange"),
        ):
            if history:
                xs, ys = zip(*history)
                ys = [to_y(v) for v in ys]
                plotted.extend(ys)
                plt.plot(xs, ys, color=color, marker="braille")
        if has_data:
            last_epoch = max(
                e for e, _ in self.train_loss_history + self.val_loss_history
            )
            step = max(1, math.ceil((last_epoch + 1) / 8))
            plt.xticks(list(range(0, last_epoch + 1, step)))
        if self.log_scale and plotted:
            positions, labels = _log_ticks(min(plotted), max(plotted))
            plt.yticks(positions, labels)
        if self.best_val_epoch is not None:
            plt.scatter(
                [self.best_val_epoch],
                [to_y(self.best_val_loss)],
                color="green",
                marker="●",
            )
        plot.refresh()

        scale = "log" if self.log_scale else "linear"
        panel = self.query_one("#loss-panel")
        panel.border_title = (
            "Loss  [cyan]━ train[/]  [#ff8700]━ val[/]  [green]● best[/]  "
            f"[dim]({scale} · press l)[/]"
        )
        parts = []
        if self.train_loss_history:
            parts.append(f"train {_fmt(self.train_loss_history[-1][1])}")
        if self.val_loss_history:
            parts.append(f"val {_fmt(self.val_loss_history[-1][1])}")
        if self.best_val_epoch is not None:
            parts.append(f"best {_fmt(self.best_val_loss)} @{self.best_val_epoch}")
        panel.border_subtitle = " · ".join(parts)

    def end_epoch(self, epoch: int, metrics: Dict[str, float], elapsed: float) -> None:
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        val_mae = metrics.get("val_mae")

        if train_loss is not None and not math.isnan(train_loss):
            self.train_loss_history.append((epoch, train_loss))

        improved = False
        if val_loss is not None and not math.isnan(val_loss):
            self.val_loss_history.append((epoch, val_loss))
            if self.best_val_loss is None or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch
                improved = True
                self._set_stat("best_val", f"{_fmt(val_loss)} @{epoch}")

        self._redraw_loss()

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
