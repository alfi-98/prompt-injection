from __future__ import annotations
import os, re
from pathlib import Path

def make_run_dir(base: str = "results") -> str:
    Path(base).mkdir(parents=True, exist_ok=True)
    pat = re.compile(r"^test_(\d{3})$")
    nums = []
    for name in os.listdir(base):
        m = pat.match(name)
        if m: nums.append(int(m.group(1)))
    nxt = (max(nums) + 1) if nums else 1
    run_dir = Path(base) / f"test_{nxt:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)
