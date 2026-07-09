# Brain Viewer

Static Three.js viewer for Advanced-SNN Roihu snapshot files.

Generate snapshots through a Roihu Slurm validation job, copy the `.ndjson` output to the workstation where the browser runs, then open `viewer/index.html` through any static file server and load the snapshot with the file button.

The viewer is not a validation target. It is for inspecting already generated Roihu outputs.

## Controls

- Load a `.ndjson` snapshot file.
- Play, pause, step, and scrub frames.
- Click a neuron to inspect the biological microscope view.
- Click a region row to apply a visual stimulation overlay.
- Use `sampled` mode for individual neuron inspection and `aggregate` mode for large Roihu outputs.

The raster panel shows recent regional spike density, and the scale panel reports sampled aggregate activity.
