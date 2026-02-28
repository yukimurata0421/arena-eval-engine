#!/usr/bin/env python3
"""
Deprecated wrapper. Use `arena run ...` instead.
"""
import sys
from arena.cli import main

if __name__ == "__main__":
    print("[WARN] Direct execution of scripts/master.py is deprecated. Use `arena run`.\n")
    sys.exit(main(["run", *sys.argv[1:]]))
