extern "C" __global__ void lif_dense_update(
    float* voltage,
    const float* input_current,
    unsigned short* refractory_left,
    unsigned char* spike_flags,
    unsigned int neurons,
    float decay,
    float threshold,
    float reset,
    unsigned short refractory_period) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= neurons) {
        return;
    }

    if (refractory_left[i] > 0) {
        refractory_left[i] -= 1;
        spike_flags[i] = 0;
        return;
    }

    float updated = voltage[i] * decay + input_current[i];
    if (updated >= threshold) {
        voltage[i] = reset;
        refractory_left[i] = refractory_period;
        spike_flags[i] = 1;
    } else {
        voltage[i] = updated;
        spike_flags[i] = 0;
    }
}
