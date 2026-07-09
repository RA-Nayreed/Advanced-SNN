#!/usr/bin/env python3
import argparse
import json
import math
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"

REGIONS = [
    {"id": 0, "name": "sensory", "center": [-0.7, -0.15, 0.05], "radius": 0.42, "color": [0.18, 0.72, 1.0]},
    {"id": 1, "name": "association", "center": [-0.15, 0.16, 0.02], "radius": 0.5, "color": [0.62, 0.92, 0.35]},
    {"id": 2, "name": "memory", "center": [0.36, -0.18, -0.08], "radius": 0.44, "color": [1.0, 0.67, 0.23]},
    {"id": 3, "name": "motor", "center": [0.78, 0.12, 0.1], "radius": 0.4, "color": [1.0, 0.28, 0.42]},
    {"id": 4, "name": "core", "center": [0.0, 0.02, -0.26], "radius": 0.34, "color": [0.72, 0.55, 1.0]},
]

class LiveBrainHandler(SimpleHTTPRequestHandler):
    server_version = "AdvancedSNNLive/0.1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(VIEWER), **kwargs)

    def log_message(self, fmt, *args):
        if not getattr(self.server, "quiet", False):
            super().log_message(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/live.ndjson":
            self.stream_live(parsed.query)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def stream_live(self, query):
        params = parse_qs(query)
        steps = int(params.get("steps", [self.server.steps])[0])
        neurons = int(params.get("neurons", [self.server.neurons])[0])
        delay = float(params.get("delay", [self.server.delay])[0])
        synapses = int(params.get("synapses", [self.server.synapses])[0])

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        for step in range(steps):
            frame = make_frame(step, neurons, synapses)
            payload = json.dumps(frame, separators=(",", ":")).encode("utf-8") + b"\n"
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except BrokenPipeError:
                break
            time.sleep(delay)

def make_frame(step, neuron_count, synapse_count):
    neurons = []
    wave = step * 0.13
    for i in range(neuron_count):
        region = REGIONS[i % len(REGIONS)]
        a = i * 2.399 + wave * 0.08
        r = ((i * 37) % 100) / 100 * region["radius"]
        z = ((((i * 19) % 100) / 100) - 0.5) * region["radius"]
        voltage = 0.35 + 0.45 * math.sin(step * 0.17 + i * 0.071)
        spiked = (i + step * 7) % 41 == 0 or (region["id"] == step % len(REGIONS) and i % 53 == 0)
        neurons.append({
            "id": i,
            "region_id": region["id"],
            "kind": "inhibitory" if i % 6 == 0 else "excitatory",
            "position": [
                region["center"][0] + math.cos(a) * r,
                region["center"][1] + math.sin(a) * r * 0.75,
                region["center"][2] + z,
            ],
            "voltage": max(0.0, min(1.0, voltage)),
            "input_current": 0.0,
            "refractory_left": 0,
            "spiked": spiked,
        })

    synapses = []
    for i in range(synapse_count):
        synapses.append({
            "source": i % neuron_count,
            "target": (i * 17 + 11) % neuron_count,
            "weight": -0.04 if i % 6 == 0 else 0.035 + 0.02 * math.sin(step * 0.05 + i * 0.01),
        })

    active = sum(1 for neuron in neurons if neuron["spiked"])
    return {
        "schema_version": 3,
        "step": step,
        "neurons_total": neuron_count,
        "synapses_total": synapse_count,
        "regions": REGIONS,
        "neurons": neurons,
        "synapses": synapses,
        "metrics": {
            "total_spikes": step * active,
            "active_input_spikes": max(0, active - 2),
            "active_output_spikes": active,
            "synapse_events_processed": step * synapse_count,
            "mean_sample_voltage": sum(neuron["voltage"] for neuron in neurons) / max(1, len(neurons)),
            "stdp_potentiated": step * 8,
            "stdp_depressed": step * 5,
            "mean_abs_weight": 0.041 + 0.006 * math.sin(step * 0.07),
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Serve the Advanced-SNN viewer with a live NDJSON stream")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5173)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--neurons", type=int, default=360)
    parser.add_argument("--synapses", type=int, default=900)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), LiveBrainHandler)
    server.steps = args.steps
    server.neurons = args.neurons
    server.synapses = args.synapses
    server.delay = args.delay
    server.quiet = args.quiet
    print(f"serving viewer at http://{args.host}:{args.port}")
    print(f"live stream at http://{args.host}:{args.port}/live.ndjson")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
