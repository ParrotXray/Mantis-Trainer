from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, SelectionList, Static
from textual.widgets.selection_list import Selection

_STAGE_OPTIONS = [
    ("Data Preprocessing", "datapreprocess"),
    ("Train LSTM Deep Autoencoder", "deepautoencoder"),
    ("Export to ONNX", "export"),
]


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
    #stage-list { height: auto; max-height: 6; border: solid $accent; }
    #set-input, #path-input { margin-top: 1; }
    #error-msg { color: $error; margin: 1 0; }
    #button-row { margin-top: 1; height: 3; }
    """

    def __init__(self):
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Label("Select pipeline stage(s) to run:")
            yield SelectionList[str](
                *[Selection(label, value) for label, value in _STAGE_OPTIONS],
                id="stage-list",
            )
            yield Label(
                "Kaggle dataset id(s), comma-separated (required if Data "
                "Preprocessing is checked), e.g. owner/dataset-name:"
            )
            yield Input(placeholder="xxx/xxx-dataset,yyy/yyy-dataset", id="set-input")
            yield Label(
                "Optional: local dataset path(s), comma-separated in the same "
                "order as the dataset ids above (overrides auto-download):"
            )
            yield Input(placeholder="/data/xxx,/data/yyy", id="path-input")
            yield Static("", id="error-msg")
            with Horizontal(id="button-row"):
                yield Button("Start", id="start-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="error")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Mantis Trainer — Launcher"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.exit(result=None)
        elif event.button.id == "start-btn":
            self._submit()

    def _submit(self) -> None:
        error = self.query_one("#error-msg", Static)
        stages = set(self.query_one("#stage-list", SelectionList).selected)

        if not stages:
            error.update("Pick at least one stage.")
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
                error.update(
                    "Data Preprocessing needs at least one Kaggle dataset id."
                )
                return
            datasets = [s.strip() for s in set_raw.split(",")]
            selection["set"] = set_raw

            path_raw = self.query_one("#path-input", Input).value.strip()
            if path_raw:
                paths = [p.strip() for p in path_raw.split(",")]
                if len(paths) != len(datasets):
                    error.update(
                        f"Path count ({len(paths)}) must match dataset "
                        f"id count ({len(datasets)})."
                    )
                    return
                selection["path"] = path_raw

        self.exit(result=selection)
