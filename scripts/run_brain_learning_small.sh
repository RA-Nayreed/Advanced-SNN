#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- learn --neurons 2000 --fanout 32 --steps 300 --seed 7 \
  --external-prob 0.006 --external-current 1.2 \
  --snapshot-out brain_learning.ndjson --snapshot-every 2 --snapshot-neurons 1500 --snapshot-synapses 4000
