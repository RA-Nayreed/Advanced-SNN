#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- gpu-dense --neurons 10000 --steps 100 --seed 1
