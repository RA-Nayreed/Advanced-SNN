import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const canvas = document.querySelector('#brain-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setClearColor(0x07080b, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07080b, 0.08);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0.2, 4.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 1.1;
controls.maxDistance = 8;

const root = new THREE.Group();
scene.add(root);

const ambient = new THREE.AmbientLight(0x7d8da8, 0.5);
scene.add(ambient);

const key = new THREE.PointLight(0x9ee6ff, 2.0, 10);
key.position.set(-2.2, 2.4, 3.5);
scene.add(key);

const fill = new THREE.PointLight(0xff8e71, 0.9, 8);
fill.position.set(2.8, -1.4, 2.2);
scene.add(fill);

const neuronGeometry = new THREE.SphereGeometry(0.018, 10, 8);
const neuronMaterial = new THREE.MeshStandardMaterial({
  color: 0x6ed6ff,
  emissive: 0x123447,
  roughness: 0.38,
  metalness: 0.05,
});
const neuronsMesh = new THREE.InstancedMesh(neuronGeometry, neuronMaterial, 1);
neuronsMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
root.add(neuronsMesh);

const synapseMaterial = new THREE.LineBasicMaterial({
  color: 0x6cb4ff,
  transparent: true,
  opacity: 0.22,
  blending: THREE.AdditiveBlending,
});
let synapseLines = new THREE.LineSegments(new THREE.BufferGeometry(), synapseMaterial);
root.add(synapseLines);

const pulseMaterial = new THREE.PointsMaterial({
  color: 0xffffff,
  size: 0.035,
  transparent: true,
  opacity: 0.8,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});
let pulsePoints = new THREE.Points(new THREE.BufferGeometry(), pulseMaterial);
root.add(pulsePoints);

const els = {
  file: document.querySelector('#snapshot-file'),
  loadSample: document.querySelector('#load-sample'),
  playPause: document.querySelector('#play-pause'),
  stepBack: document.querySelector('#step-back'),
  stepForward: document.querySelector('#step-forward'),
  slider: document.querySelector('#frame-slider'),
  speed: document.querySelector('#speed-select'),
  frameCount: document.querySelector('#frame-count'),
  step: document.querySelector('#step-value'),
  spikes: document.querySelector('#spike-value'),
  events: document.querySelector('#event-value'),
  neurons: document.querySelector('#neurons-value'),
  synapses: document.querySelector('#synapses-value'),
  active: document.querySelector('#active-value'),
  voltage: document.querySelector('#voltage-value'),
  regions: document.querySelector('#region-list'),
  status: document.querySelector('#status-line'),
};

let frames = [];
let frameIndex = 0;
let playing = false;
let lastAdvance = 0;
let neuronIdToIndex = new Map();
let positionsById = new Map();
const scratchMatrix = new THREE.Matrix4();
const scratchColor = new THREE.Color();

function parseNdjson(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function loadSnapshotText(text, label) {
  frames = parseNdjson(text);
  if (!frames.length) {
    throw new Error('snapshot has no frames');
  }
  frameIndex = 0;
  playing = false;
  els.playPause.textContent = '▶';
  els.slider.max = String(frames.length - 1);
  els.slider.value = '0';
  els.frameCount.textContent = `${frames.length} frames`;
  els.status.textContent = label;
  buildStaticGeometry(frames[0]);
  applyFrame(frames[0]);
}

function buildStaticGeometry(frame) {
  neuronIdToIndex = new Map();
  positionsById = new Map();
  frame.neurons.forEach((neuron, index) => {
    neuronIdToIndex.set(neuron.id, index);
    positionsById.set(neuron.id, new THREE.Vector3(...neuron.position));
  });

  neuronsMesh.count = frame.neurons.length;
  neuronsMesh.instanceMatrix.needsUpdate = true;
  if (neuronsMesh.instanceColor) {
    neuronsMesh.instanceColor.needsUpdate = true;
  }

  const linePositions = [];
  for (const synapse of frame.synapses || []) {
    const source = positionsById.get(synapse.source);
    const target = positionsById.get(synapse.target);
    if (!source || !target) {
      continue;
    }
    linePositions.push(source.x, source.y, source.z, target.x, target.y, target.z);
  }
  synapseLines.geometry.dispose();
  synapseLines.geometry = new THREE.BufferGeometry();
  synapseLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));

  renderRegions(frame.regions || []);
}

function applyFrame(frame) {
  const spiked = new Set(frame.neurons.filter((neuron) => neuron.spiked).map((neuron) => neuron.id));

  frame.neurons.forEach((neuron, index) => {
    const radius = neuron.spiked ? 0.046 : neuron.kind === 'inhibitory' ? 0.024 : 0.029;
    const intensity = Math.min(1, Math.max(0, neuron.voltage));
    scratchMatrix.makeScale(radius, radius, radius);
    scratchMatrix.setPosition(neuron.position[0], neuron.position[1], neuron.position[2]);
    neuronsMesh.setMatrixAt(index, scratchMatrix);

    if (neuron.spiked) {
      scratchColor.setRGB(1.0, 0.96, 0.72);
    } else if (neuron.kind === 'inhibitory') {
      scratchColor.setRGB(1.0, 0.34 + intensity * 0.25, 0.42);
    } else {
      scratchColor.setRGB(0.24 + intensity * 0.55, 0.72, 1.0);
    }
    neuronsMesh.setColorAt(index, scratchColor);
  });
  neuronsMesh.instanceMatrix.needsUpdate = true;
  neuronsMesh.instanceColor.needsUpdate = true;

  const pulsePositions = [];
  for (const synapse of frame.synapses || []) {
    if (!spiked.has(synapse.source)) {
      continue;
    }
    const source = positionsById.get(synapse.source);
    const target = positionsById.get(synapse.target);
    if (!source || !target) {
      continue;
    }
    const pulse = source.clone().lerp(target, 0.45 + 0.2 * Math.sin(performance.now() * 0.008));
    pulsePositions.push(pulse.x, pulse.y, pulse.z);
  }
  pulsePoints.geometry.dispose();
  pulsePoints.geometry = new THREE.BufferGeometry();
  pulsePoints.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pulsePositions, 3));

  els.step.textContent = String(frame.step ?? frameIndex);
  els.spikes.textContent = String(frame.metrics?.total_spikes ?? 0);
  els.events.textContent = String(frame.metrics?.synapse_events_processed ?? 0);
  els.neurons.textContent = String(frame.neurons_total ?? frame.neurons.length);
  els.synapses.textContent = String(frame.synapses_total ?? (frame.synapses || []).length);
  els.active.textContent = String(frame.metrics?.active_output_spikes ?? 0);
  els.voltage.textContent = Number(frame.metrics?.mean_sample_voltage ?? 0).toFixed(3);
}

function renderRegions(regions) {
  els.regions.textContent = '';
  for (const region of regions) {
    const row = document.createElement('div');
    row.className = 'region-row';
    const swatch = document.createElement('i');
    swatch.className = 'swatch';
    const color = region.color || [0.4, 0.7, 1.0];
    swatch.style.background = `rgb(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)})`;
    const name = document.createElement('span');
    name.textContent = region.name || `region ${region.id}`;
    const radius = document.createElement('strong');
    radius.textContent = Number(region.radius || 0).toFixed(2);
    row.append(swatch, name, radius);
    els.regions.append(row);
  }
}

function selectFrame(index) {
  if (!frames.length) {
    return;
  }
  frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  els.slider.value = String(frameIndex);
  applyFrame(frames[frameIndex]);
}

function resize() {
  const { clientWidth, clientHeight } = canvas;
  renderer.setSize(clientWidth, clientHeight, false);
  camera.aspect = clientWidth / Math.max(1, clientHeight);
  camera.updateProjectionMatrix();
}

function animate(now) {
  requestAnimationFrame(animate);
  resize();
  controls.update();
  root.rotation.y += 0.0015;

  if (playing && frames.length && now - lastAdvance > Number(els.speed.value)) {
    selectFrame((frameIndex + 1) % frames.length);
    lastAdvance = now;
  }

  renderer.render(scene, camera);
}

els.file.addEventListener('change', async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  try {
    await loadSnapshotText(await file.text(), file.name);
  } catch (error) {
    els.status.textContent = error.message;
  }
});

els.loadSample.addEventListener('click', async () => {
  try {
    const response = await fetch('sample.ndjson');
    if (!response.ok) {
      throw new Error(`sample unavailable: ${response.status}`);
    }
    await loadSnapshotText(await response.text(), 'sample.ndjson');
  } catch (error) {
    els.status.textContent = error.message;
  }
});

els.playPause.addEventListener('click', () => {
  playing = !playing;
  els.playPause.textContent = playing ? 'Ⅱ' : '▶';
});

els.stepBack.addEventListener('click', () => selectFrame(frameIndex - 1));
els.stepForward.addEventListener('click', () => selectFrame(frameIndex + 1));
els.slider.addEventListener('input', () => selectFrame(Number(els.slider.value)));

loadSnapshotText(`
${sampleFrame(0, 0)}
${sampleFrame(1, 8)}
${sampleFrame(2, 14)}
${sampleFrame(3, 21)}
${sampleFrame(4, 26)}
`, 'generated sample');
requestAnimationFrame(animate);

function sampleFrame(step, offset) {
  const regions = [
    { id: 0, name: 'sensory', center: [-0.7, -0.15, 0.05], radius: 0.42, color: [0.18, 0.72, 1.0] },
    { id: 1, name: 'association', center: [-0.15, 0.16, 0.02], radius: 0.5, color: [0.62, 0.92, 0.35] },
    { id: 2, name: 'memory', center: [0.36, -0.18, -0.08], radius: 0.44, color: [1.0, 0.67, 0.23] },
    { id: 3, name: 'motor', center: [0.78, 0.12, 0.1], radius: 0.4, color: [1.0, 0.28, 0.42] },
  ];
  const neurons = [];
  for (let i = 0; i < 180; i += 1) {
    const region = regions[i % regions.length];
    const a = i * 2.399 + step * 0.025;
    const r = ((i * 37) % 100) / 100 * region.radius;
    const z = ((((i * 19) % 100) / 100) - 0.5) * region.radius;
    neurons.push({
      id: i,
      region_id: region.id,
      kind: i % 6 === 0 ? 'inhibitory' : 'excitatory',
      position: [
        region.center[0] + Math.cos(a) * r,
        region.center[1] + Math.sin(a) * r * 0.75,
        region.center[2] + z,
      ],
      voltage: ((i + step * 7) % 30) / 30,
      input_current: 0,
      refractory_left: 0,
      spiked: (i + offset) % 29 === 0,
    });
  }
  const synapses = [];
  for (let i = 0; i < 280; i += 1) {
    synapses.push({ source: i % 180, target: (i * 17 + 11) % 180, weight: i % 6 === 0 ? -0.04 : 0.04 });
  }
  return JSON.stringify({
    schema_version: 2,
    step,
    neurons_total: 180,
    synapses_total: 280,
    regions,
    neurons,
    synapses,
    metrics: {
      total_spikes: step * 21,
      active_input_spikes: Math.max(0, step * 4),
      active_output_spikes: 7 + step,
      synapse_events_processed: step * 420,
      mean_sample_voltage: 0.32 + step * 0.03,
    },
  });
}
