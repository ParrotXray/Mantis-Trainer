from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessConfig:
    clip_min: float = -1e9
    clip_max: float = 1e9
    fill_value: float = 0.0
    # Rows whose Mantis "ML Verdict" is one of these are dropped at load time.
    # By default only anomalies backed by a Suricata rule match:
    # confirmed_attack (rule match plus an aggregator FLOOD / SCAN / TARGETED
    # verdict) and rule_match (rule match alone). "structural" (aggregator
    # verdict alone) is kept: those heuristics also flag a busy client's
    # ordinary egress. Dropping plain "anomaly" would trim the benign tail on
    # every retrain, since the live threshold flags a fixed share of benign
    # traffic by definition.
    exclude_verdicts: Tuple[str, ...] = ("confirmed_attack", "rule_match")
    # Rows whose source IP is in one of these addresses or CIDR blocks are
    # dropped at load time, which removes those hosts' whole sequences. Meant
    # for traffic the model must not learn as normal (a compromised host, a
    # pentest box, a scanner), not for hosts that alert too often: a host the
    # model never saw scores higher at inference, so those belong in a Mantis
    # suppression rule instead.
    exclude_src_ips: Tuple[str, ...] = ()
    # Same for the destination IP. This also drops other hosts' flows to these
    # addresses from the middle of their sequences, while Mantis still sees
    # them live, so prefer exclude_src_ips.
    exclude_dst_ips: Tuple[str, ...] = ()
