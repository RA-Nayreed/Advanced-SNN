#!/usr/bin/env bash
set -euo pipefail

cargo run --release -- brain --neurons 2000 --fanout 32 --steps 200 --seed 7 \
  --snapshot-out brain_blob.ndjson --snapshot-every 2 --snapshot-neurons 1500 --snapshot-synapses 4000
