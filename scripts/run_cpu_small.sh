#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- cpu --neurons 1000 --fanout 32 --steps 100 --seed 1
