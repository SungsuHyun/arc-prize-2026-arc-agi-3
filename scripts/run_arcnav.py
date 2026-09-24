#!/usr/bin/env python
"""Thin CLI for the arcnav agent: python scripts/run_arcnav.py --games ls20,vc33 --minutes 20 --jobs 2 --tag smoke"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcnav.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
