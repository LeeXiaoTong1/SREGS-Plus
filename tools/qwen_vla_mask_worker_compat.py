#!/usr/bin/env python3
"""Compatibility wrapper for qwen_vla_mask_worker.py.

The VLA controller may pass controller-side options that the original Qwen
worker does not need. This wrapper removes those options and then executes the
original worker.
"""

import runpy
import sys


def _strip_flag(argv, flag, takes_value=False):
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if takes_value and i + 1 < len(argv) else 1
            continue
        out.append(argv[i])
        i += 1
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    args = _strip_flag(args, "--force_regions", takes_value=False)
    args = _strip_flag(args, "--min_regions", takes_value=True)
    sys.argv = ["tools/qwen_vla_mask_worker.py"] + args
    runpy.run_path("tools/qwen_vla_mask_worker.py", run_name="__main__")
