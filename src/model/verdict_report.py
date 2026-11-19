from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Tuple

# Every value Mantis writes in the "ML Verdict" column. Anything else in a CSV
# is reported as unknown (a newer Mantis, or a damaged file).
ML_VERDICTS: Final[Tuple[str, ...]] = (
    "normal",
    "anomaly",
    "structural",
    "rule_match",
    "confirmed_attack",
    "suppressed",
    "whitelisted",
    "uninspected",
)

# Verdicts backed by a Suricata rule match, the ones dp.exclude_verdicts drops
# by default.
RULE_BACKED_VERDICTS: Final[Tuple[str, ...]] = ("confirmed_attack", "rule_match")

# Rows an operator confirmed as benign. The report never counts them as rows
# that excluding a flagged source would drop: they are what a retrain should
# learn as normal.
OPERATOR_BENIGN_VERDICTS: Final[Tuple[str, ...]] = ("suppressed",)

VERDICT_REPORT_WINDOW_SECS: Final[Tuple[int, int]] = (0, 86_400)
VERDICT_REPORT_TOP: Final[Tuple[int, int]] = (1, 1_000)


@dataclass(frozen=True)
class VerdictReportConfig:
    # A source is flagged when it has at least one row with one of these
    # verdicts.
    trigger_verdicts: Tuple[str, ...] = RULE_BACKED_VERDICTS
    # Half-width of the window around each trigger row used for the
    # "within window" estimate; 0 counts only rows at the same second.
    window_secs: int = 600
    # Flagged sources listed, most trigger rows first.
    top: int = 50
    chunk_rows: int = 500_000


@dataclass
class VerdictReportSource:
    src_ip: str
    private: bool
    trigger_rows: int
    total_rows: int
    verdicts: Dict[str, int]
    first_trigger_ms: Optional[int]
    last_trigger_ms: Optional[int]
    # Rows of this source, other than the trigger rows and operator-confirmed
    # rows, that excluding it would drop: in the whole data, or only within
    # window_secs of one of its trigger rows.
    all_extra_rows: int = 0
    window_extra_rows: int = 0


@dataclass
class VerdictReportResult:
    files: int
    files_without_verdict: List[str]
    rows: int
    verdicts: Dict[str, int]
    unknown_verdicts: List[str]
    trigger_verdicts: List[str]
    window_secs: int
    trigger_rows: int
    flagged_sources: int
    all_extra_rows: int
    window_extra_rows: int
    unparsed_timestamps: int
    sources: List[VerdictReportSource] = field(default_factory=list)
    truncated: bool = False
