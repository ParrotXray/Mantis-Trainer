import ipaddress
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from model import (
    ML_VERDICT_COLUMN,
    ML_VERDICTS,
    OPERATOR_BENIGN_VERDICTS,
    VERDICT_REPORT_TOP,
    VERDICT_REPORT_WINDOW_SECS,
    ReportError,
    VerdictReportConfig,
    VerdictReportResult,
    VerdictReportSource,
)

SOURCE_IP_COLUMNS: Final[Tuple[str, ...]] = ("Source IP", "Src IP")
TIMESTAMP_COLUMN: Final[str] = "Timestamp"
# Mantis writes the flow start in UTC with this format (flow_features.rs).
MANTIS_TIMESTAMP_FORMAT: Final[str] = "%m/%d/%Y %H:%M:%S"


class VerdictReport:
    """Per-source breakdown of Mantis "ML Verdict" values in a set of CSVs.

    Shows how much data dp.exclude_verdicts drops and what excluding the
    sources it flags (dp.exclude_src_ips) would cost, in the whole data or
    only around each flagged row, before deciding on either.
    """

    def __init__(self, config: VerdictReportConfig) -> None:
        self.config = config
        self.validate(config)

    @staticmethod
    def validate(config: VerdictReportConfig) -> None:
        if not config.trigger_verdicts:
            raise ReportError("At least one trigger verdict is required")
        unknown = [v for v in config.trigger_verdicts if v not in ML_VERDICTS]
        if unknown:
            raise ReportError(
                f"Unknown verdict(s) {unknown}; expected one of {list(ML_VERDICTS)}"
            )
        low, high = VERDICT_REPORT_WINDOW_SECS
        if not low <= config.window_secs <= high:
            raise ReportError(f"window_secs must be within {low}..{high}")
        low, high = VERDICT_REPORT_TOP
        if not low <= config.top <= high:
            raise ReportError(f"top must be within {low}..{high}")

    @staticmethod
    def _columns(path: Path) -> Dict[str, str]:
        header = pd.read_csv(path, nrows=0, encoding="utf-8", encoding_errors="replace")
        return {str(name).strip(): str(name) for name in header.columns}

    def _chunks(self, path: Path, columns: Sequence[str]) -> Iterator[pd.DataFrame]:
        yield from pd.read_csv(
            path,
            usecols=list(columns),
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
            encoding_errors="replace",
            chunksize=self.config.chunk_rows,
        )

    @staticmethod
    def timestamps_ms(series: pd.Series) -> np.ndarray:
        ts = pd.to_datetime(series, format=MANTIS_TIMESTAMP_FORMAT, errors="coerce")
        if len(series) and ts.isna().mean() > 0.5:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = pd.to_datetime(series, errors="coerce", utc=True)
            ts = ts.dt.tz_localize(None)
        values = ts.to_numpy(dtype="datetime64[ms]").astype("int64")
        return np.where(ts.isna().to_numpy(), np.iinfo(np.int64).min, values)

    @staticmethod
    def _is_private(ip: str) -> bool:
        try:
            return ipaddress.ip_address(ip).is_private
        except ValueError:
            return False

    def build(
        self, paths: Sequence[Path], root: Optional[Path] = None
    ) -> VerdictReportResult:
        triggers: Set[str] = set(self.config.trigger_verdicts)
        verdicts: Counter = Counter()
        per_source: Dict[str, Counter] = defaultdict(Counter)
        trigger_times: Dict[str, List[np.ndarray]] = defaultdict(list)
        without_verdict: List[str] = []
        readable: List[Tuple[Path, str, Optional[str], Optional[str]]] = []
        rows = 0
        unparsed = 0

        for path in paths:
            names = self._columns(path)
            verdict_col = names.get(ML_VERDICT_COLUMN)
            if verdict_col is None:
                without_verdict.append(
                    str(path.relative_to(root)) if root else path.name
                )
                continue
            src_col = next((names[c] for c in SOURCE_IP_COLUMNS if c in names), None)
            ts_col = names.get(TIMESTAMP_COLUMN)
            readable.append((path, verdict_col, src_col, ts_col))
            columns = [c for c in (verdict_col, src_col, ts_col) if c is not None]
            for chunk in self._chunks(path, columns):
                verdict = chunk[verdict_col].str.strip()
                rows += len(chunk)
                verdicts.update(verdict.value_counts().to_dict())
                if src_col is None:
                    continue
                src = chunk[src_col].str.strip()
                for (ip, value), count in (
                    pd.DataFrame({"ip": src, "v": verdict})
                    .groupby(["ip", "v"], sort=False)
                    .size()
                    .items()
                ):
                    per_source[ip][value] += int(count)
                hit = verdict.isin(triggers).to_numpy()
                if ts_col is None or not hit.any():
                    continue
                times = self.timestamps_ms(chunk[ts_col][hit])
                hit_src = src[hit].to_numpy()
                valid = times != np.iinfo(np.int64).min
                unparsed += int((~valid).sum())
                for ip in np.unique(hit_src[valid]):
                    trigger_times[ip].append(times[valid & (hit_src == ip)])

        flagged = {
            ip: counts
            for ip, counts in per_source.items()
            if any(counts.get(v, 0) for v in triggers)
        }
        windows = self._window_rows(readable, flagged, trigger_times, triggers)

        sources: List[VerdictReportSource] = []
        for ip, counts in flagged.items():
            trigger_rows = sum(counts.get(v, 0) for v in triggers)
            benign = sum(counts.get(v, 0) for v in OPERATOR_BENIGN_VERDICTS)
            total = sum(counts.values())
            times = np.concatenate(trigger_times[ip]) if trigger_times.get(ip) else None
            sources.append(
                VerdictReportSource(
                    src_ip=ip,
                    private=self._is_private(ip),
                    trigger_rows=trigger_rows,
                    total_rows=total,
                    verdicts=dict(counts.most_common()),
                    first_trigger_ms=int(times.min()) if times is not None else None,
                    last_trigger_ms=int(times.max()) if times is not None else None,
                    all_extra_rows=total - trigger_rows - benign,
                    window_extra_rows=windows.get(ip, 0),
                )
            )
        sources.sort(key=lambda s: (-s.trigger_rows, -s.total_rows, s.src_ip))

        return VerdictReportResult(
            files=len(paths),
            files_without_verdict=without_verdict,
            rows=rows,
            verdicts=dict(verdicts.most_common()),
            unknown_verdicts=sorted(v for v in verdicts if v not in ML_VERDICTS),
            trigger_verdicts=list(self.config.trigger_verdicts),
            window_secs=self.config.window_secs,
            trigger_rows=sum(verdicts.get(v, 0) for v in triggers),
            flagged_sources=len(sources),
            all_extra_rows=sum(s.all_extra_rows for s in sources),
            window_extra_rows=sum(s.window_extra_rows for s in sources),
            unparsed_timestamps=unparsed,
            sources=sources[: self.config.top],
            truncated=len(sources) > self.config.top,
        )

    def _window_rows(
        self,
        readable: Sequence[Tuple[Path, str, Optional[str], Optional[str]]],
        flagged: Dict[str, Counter],
        trigger_times: Dict[str, List[np.ndarray]],
        triggers: Set[str],
    ) -> Dict[str, int]:
        sorted_times = {
            ip: np.sort(np.concatenate(parts))
            for ip, parts in trigger_times.items()
            if ip in flagged and parts
        }
        counts: Counter = Counter()
        if not sorted_times:
            return {}
        skip = triggers | set(OPERATOR_BENIGN_VERDICTS)
        window_ms = self.config.window_secs * 1000
        for path, verdict_col, src_col, ts_col in readable:
            if src_col is None or ts_col is None:
                continue
            for chunk in self._chunks(path, (verdict_col, src_col, ts_col)):
                src = chunk[src_col].str.strip()
                keep = (
                    src.isin(sorted_times.keys())
                    & ~chunk[verdict_col].str.strip().isin(skip)
                ).to_numpy()
                if not keep.any():
                    continue
                ips = src[keep].to_numpy()
                times = self.timestamps_ms(chunk[ts_col][keep])
                for ip in np.unique(ips):
                    marks = sorted_times[ip]
                    at = times[ips == ip]
                    at = at[at != np.iinfo(np.int64).min]
                    if not len(at):
                        continue
                    idx = np.searchsorted(marks, at)
                    before = np.abs(at - marks[np.clip(idx - 1, 0, len(marks) - 1)])
                    after = np.abs(marks[np.clip(idx, 0, len(marks) - 1)] - at)
                    counts[ip] += int((np.minimum(before, after) <= window_ms).sum())
        return dict(counts)

    @staticmethod
    def _time(ms: Optional[int]) -> str:
        if ms is None:
            return "-"
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    @staticmethod
    def _share(part: int, whole: int) -> str:
        return f"{part / whole:.3%}" if whole else "-"

    @classmethod
    def render(cls, result: VerdictReportResult) -> str:
        lines = [
            f"Files: {result.files} ({len(result.files_without_verdict)} without "
            f"{ML_VERDICT_COLUMN})",
            f"Rows:  {result.rows:,}",
            "",
            f"{ML_VERDICT_COLUMN} distribution:",
        ]
        for verdict, count in result.verdicts.items():
            mark = "  (unknown)" if verdict in result.unknown_verdicts else ""
            lines.append(
                f"  {verdict or '(empty)':<18} {count:>12,}  "
                f"{cls._share(count, result.rows):>9}{mark}"
            )
        lines += [
            "",
            f"Trigger verdicts: {', '.join(result.trigger_verdicts)}",
            f"Trigger rows (dropped by dp.exclude_verdicts): {result.trigger_rows:,} "
            f"({cls._share(result.trigger_rows, result.rows)})",
            f"Flagged sources: {result.flagged_sources}",
            f"Other rows of flagged sources, suppressed excluded:",
            f"  whole data:            {result.all_extra_rows:>12,}  "
            f"({cls._share(result.all_extra_rows, result.rows)})",
            f"  within ±{result.window_secs}s of a trigger: "
            f"{result.window_extra_rows:>12,}  "
            f"({cls._share(result.window_extra_rows, result.rows)})",
        ]
        if result.unparsed_timestamps:
            lines.append(
                f"Trigger rows with an unreadable Timestamp (left out of the "
                f"window estimate): {result.unparsed_timestamps:,}"
            )
        if result.sources:
            lines += [
                "",
                f"  {'Source IP':<40} {'Scope':<8} {'Trigger':>8} {'Rows':>10} "
                f"{'Share':>8} {'Window':>8}  First / last trigger (UTC)",
            ]
            for s in result.sources:
                lines.append(
                    f"  {s.src_ip:<40} {'private' if s.private else 'public':<8} "
                    f"{s.trigger_rows:>8,} {s.total_rows:>10,} "
                    f"{cls._share(s.total_rows, result.rows):>8} "
                    f"{s.window_extra_rows:>8,}  "
                    f"{cls._time(s.first_trigger_ms)} / {cls._time(s.last_trigger_ms)}"
                )
            if result.truncated:
                lines.append(
                    f"  … {result.flagged_sources - len(result.sources)} more "
                    f"(raise --top)"
                )
        return "\n".join(lines)
