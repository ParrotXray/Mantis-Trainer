from typing import Dict, List, Optional

from pydantic import BaseModel


class VerdictSource(BaseModel):
    src_ip: str
    private: bool
    trigger_rows: int
    total_rows: int
    verdicts: Dict[str, int]
    first_trigger_ms: Optional[int]
    last_trigger_ms: Optional[int]
    all_extra_rows: int
    window_extra_rows: int


class VerdictReportResponse(BaseModel):
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
    sources: List[VerdictSource]
    truncated: bool
    generated_at: float
    cached: bool
