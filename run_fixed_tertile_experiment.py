# -*- coding: utf-8 -*-
"""Run the fixed-tertile decision experiment and paired bootstrap."""

import subprocess
import sys

COMMANDS = [
    [sys.executable, "src/experiments/run_fixed_tertile_2class.py"],
    [sys.executable, "src/experiments/bootstrap_fixed_tertile_composite_vs_war.py"],
]

for command in COMMANDS:
    print("\n>>>", " ".join(command))
    subprocess.run(command, check=True)

print("\nFixed-tertile decision experiment finished.")
