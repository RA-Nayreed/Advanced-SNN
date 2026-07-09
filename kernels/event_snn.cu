#include <stdint.h>

static __device__ __forceinline__ uint64_t splitmix64_device(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    uint64_t mixed = value;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    return mixed ^ (mixed >> 31);
}

static __device__ __forceinline__ bool external_applies_device(
    uint64_t seed,
    uint64_t step,
    uint32_t neuron_id,
    float probability) {
    if (probability <= 0.0f) {
        return false;
    }
    if (probability >= 1.0f) {
        return true;
    }

    uint64_t key = seed
        ^ (step * 0x9e3779b97f4a7c15ULL)
        ^ ((uint64_t)neuron_id * 0xbf58476d1ce4e5b9ULL);
    uint64_t bits = splitmix64_device(key) >> 40;
    float unit = (float)bits * (1.0f / 16777216.0f);
    return unit < probability;
}

extern "C" __global__ void clear_input_kernel(float* input_current, uint32_t neurons) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < neurons) {
        input_current[i] = 0.0f;
    }
}

extern "C" __global__ void apply_external_input_kernel(
    float* input_current,
    uint32_t neurons,
    uint64_t seed,
    uint64_t step,
    float external_prob,
    float external_current) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < neurons && external_applies_device(seed, step, i, external_prob)) {
        input_current[i] += external_current;
    }
}

extern "C" __global__ void process_spikes_kernel(
    const uint32_t* row_ptr,
    const uint32_t* targets,
    const float* weights,
    const uint32_t* active_spikes,
    const uint32_t* active_count,
    float* input_current,
    unsigned long long* events_processed) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t count = active_count[0];
    if (i >= count) {
        return;
    }

    uint32_t source = active_spikes[i];
    uint32_t start = row_ptr[source];
    uint32_t end = row_ptr[source + 1];
    for (uint32_t edge = start; edge < end; ++edge) {
        atomicAdd(&input_current[targets[edge]], weights[edge]);
    }
    atomicAdd(events_processed, (unsigned long long)(end - start));
}

extern "C" __global__ void update_neurons_kernel(
    float* voltage,
    const float* input_current,
    unsigned short* refractory_left,
    uint32_t* next_spikes,
    uint32_t* next_count,
    uint32_t neurons,
    float decay,
    float threshold,
    float reset,
    unsigned short refractory_period) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= neurons) {
        return;
    }

    if (refractory_left[i] > 0) {
        refractory_left[i] -= 1;
        return;
    }

    float updated = voltage[i] * decay + input_current[i];
    if (updated >= threshold) {
        voltage[i] = reset;
        refractory_left[i] = refractory_period;
        uint32_t index = atomicAdd(next_count, 1U);
        next_spikes[index] = i;
    } else {
        voltage[i] = updated;
    }
}
