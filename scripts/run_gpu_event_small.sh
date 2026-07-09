#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- gpu-event --neurons 100000 --fanout 64 --steps 100 --seed 1
