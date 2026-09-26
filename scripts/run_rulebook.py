#!/usr/bin/env python
"""CLI for the rulebook agent: python scripts/run_rulebook.py --games ls20,tn36 --minutes 20 --jobs 2 --tag x [--no-model]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rulebook.run import main  # noqa: E402

if __name__ == "__main__":
    main()
