from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, SelectionList, Static
from textual.widgets.selection_list import Selection

_STAGE_OPTIONS = [
    ("Data Preprocessing", "datapreprocess"),
    ("Train LSTM Deep Autoencoder", "deepautoencoder"),
    ("Export to ONNX", "export"),
]

_BANNER = """\
█▀▄▀█ ▄▀█ █▄ █ ▀█▀ █ █▀
█ ▀ █ █▀█ █ ▀█  █  █ ▄█"""


class LauncherWizard(App):
    """Interactive stage + data-source picker shown when main.py is run
    with no stage flags at all. Returns a selection dict via App.run()'s
    result (None if the user cancels), which main.py maps onto the same
    argparse.Namespace fields the CLI flags would have set — the rest of
    the pipeline runs exactly as if those flags had been passed.

    There's no fixed catalog of datasets to pick from — get_dataset_config()
    accepts any Kaggle dataset id and only registers it on first use, so the
    dataset field here is free text, matching -s/--set on the CLI, not a
    selection list.
    """

    CSS = """
    Screen { align: center top; background: $background; }
    #scroll { width: 100%; height: 1fr; }
    #card {
        width: 90;
        max-width: 100%;
        height: auto;
        margin: 1 2;
        padding: 1 2;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        background: $surface;
    }
    #banner { width: 100%; content-align: center middle; color: $accent; text-style: bold; }
    #subtitle { width: 100%; content-align: center middle; color: $text-muted; margin-bottom: 1; }
    .panel {
        height: auto;
        margin-top: 1;
        padding: 0 1;
        border: round $panel-lighten-2;
        border-title-color: $text;
        border-title-style: bold;
        border-subtitle-color: $text-muted;
    }
    .panel:focus-within { border: round $accent; border-title-color: $accent; }
    .hint { color: $text-muted; margin-bottom: 1; }
    #stage-list { height: auto; border: none; padding: 0; background: transparent; }
    #error-msg { height: auto; margin-top: 1; color: $error; text-style: bold; display: none; }
    #error-msg.visible { display: block; }
    #button-row { height: auto; margin-top: 1; align-horizontal: right; }
    #button-row Button { margin-left: 2; min-width: 14; }
    """

    BINDINGS = [
        Binding("ctrl+s", "start", "Start"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self):
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="scroll"):
            with Center():
                with Vertical(id="card") as card:
                    card.border_title = "Launcher"
                    yield Static(_BANNER, id="banner")
                    yield Static("Anomaly-detection training pipeline", id="subtitle")

                    with Vertical(classes="panel") as stages:
                        stages.border_title = "① Pipeline stages"
                        stages.border_subtitle = "space to toggle"
                        yield SelectionList[str](
                            *[
                                Selection(label, value)
                                for label, value in _STAGE_OPTIONS
                            ],
                            id="stage-list",
                        )

                    with Vertical(classes="panel") as datasets:
                        datasets.border_title = "② Kaggle dataset id(s)"
                        datasets.border_subtitle = "required for preprocessing"
                        yield Static(
                            "Comma-separated, e.g. [b]owner/dataset-name[/b]",
                            classes="hint",
                        )
                        yield Input(
                            placeholder="xxx/xxx-dataset,yyy/yyy-dataset",
                            id="set-input",
                        )

                    with Vertical(classes="panel") as paths:
                        paths.border_title = "③ Local dataset path(s)"
                        paths.border_subtitle = "optional"
                        yield Static(
                            "Same order as the dataset ids above — "
                            "overrides auto-download",
                            classes="hint",
                        )
                        yield Input(placeholder="/data/xxx,/data/yyy", id="path-input")

                    yield Static("", id="error-msg")
                    with Horizontal(id="button-row"):
                        yield Button("Cancel", id="cancel-btn", variant="default")
                        yield Button("▶ Start", id="start-btn", variant="success")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mantis Trainer"
        self.sub_title = "Launcher"

    def action_start(self) -> None:
        self._submit()

    def action_cancel(self) -> None:
        self.exit(result=None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.exit(result=None)
        elif event.button.id == "start-btn":
            self._submit()

    def _show_error(self, message: str) -> None:
        error = self.query_one("#error-msg", Static)
        error.update(f"✖ {message}")
        error.set_class(bool(message), "visible")

    def _submit(self) -> None:
        self._show_error("")
        stages = set(self.query_one("#stage-list", SelectionList).selected)

        if not stages:
            self._show_error("Pick at least one stage.")
            return

        selection: Dict[str, Optional[str]] = {
            "datapreprocess": "datapreprocess" in stages,
            "deepautoencoder": "deepautoencoder" in stages,
            "export": "export" in stages,
            "set": None,
            "path": None,
        }

        if selection["datapreprocess"]:
            set_raw = self.query_one("#set-input", Input).value.strip()
            if not set_raw:
                self._show_error(
                    "Data Preprocessing needs at least one Kaggle dataset id."
                )
                return
            datasets = [s.strip() for s in set_raw.split(",")]
            selection["set"] = set_raw

            path_raw = self.query_one("#path-input", Input).value.strip()
            if path_raw:
                paths = [p.strip() for p in path_raw.split(",")]
                if len(paths) != len(datasets):
                    self._show_error(
                        f"Path count ({len(paths)}) must match dataset "
                        f"id count ({len(datasets)})."
                    )
                    return
                selection["path"] = path_raw

        self.exit(result=selection)
