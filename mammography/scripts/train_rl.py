#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from mammography.rl.train import main

if __name__ == "__main__":
    sys.exit(main())
