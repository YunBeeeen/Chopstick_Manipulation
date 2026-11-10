#!/usr/bin/env python3
"""Inspect the real observation layout and decode a synthetic residual action."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Deploy.common.policy_contract import (
    ACTION_DIM,
    COMMAND_TARGET_LIMITS,
    OBSERVATION_DIM,
    OBSERVATION_SLICES,
    POLICY_DT,
)
from Deploy.policy.action_adapter import decode_policy_action


def main():
    q_current = COMMAND_TARGET_LIMITS.mean(axis=1).astype(np.float32)
    raw_action = np.linspace(-1.5, 1.5, ACTION_DIM, dtype=np.float32)
    decoded = decode_policy_action(q_current, raw_action)
    print(json.dumps({
        "observation_dim": OBSERVATION_DIM,
        "observation_terms": {name: term.dim for name, term in OBSERVATION_SLICES.items()},
        "action_dim": ACTION_DIM,
        "policy_hz": round(1.0 / POLICY_DT),
        "target_shape": list(decoded.position_target.shape),
        "clipped_action_count": int(decoded.action_was_clipped.sum()),
        "clamped_target_count": int(decoded.target_was_clamped.sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
