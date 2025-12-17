#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>

#define RESERVOIR_K 3
#define RGB_DIFF_THRESHOLD 30
#define MAX_WEIGHT 50
#define MORPH_RADIUS 3
#define HYST_LOW 4
#define HYST_HIGH 30

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

struct Reservoir {
    rgb8 color;
    uint16_t weight;
};

struct GpuContext {
    int width = 0;
    int height = 0;
    Reservoir* reservoir_state = nullptr;
    uint8_t* diff_map = nullptr;
    uint8_t* morph_temp = nullptr;
    uint8_t* morph_dest = nullptr;
    uint8_t* final_mask = nullptr;
    curandState* rand_states = nullptr;
    
    void resize(int w, int h) {
        if (width == w && height == h) return;
        
        // Free old buffers
        if (reservoir_state) CUDA_CHECK(cudaFree(reservoir_state));
        if (diff_map) CUDA_CHECK(cudaFree(diff_map));
        if (morph_temp) CUDA_CHECK(cudaFree(morph_temp));
        if (morph_dest) CUDA_CHECK(cudaFree(morph_dest));
        if (final_mask) CUDA_CHECK(cudaFree(final_mask));
        if (rand_states) CUDA_CHECK(cudaFree(rand_states));
        
        width = w;
        height = h;
        
        CUDA_CHECK(cudaMalloc(&reservoir_state, w * h * RESERVOIR_K * sizeof(Reservoir)));
        CUDA_CHECK(cudaMalloc(&diff_map, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&morph_temp, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&morph_dest, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&final_mask, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&rand_states, w * h * sizeof(curandState)));
        
        // Initialize reservoirs to zero
        CUDA_CHECK(cudaMemset(reservoir_state, 0, w * h * RESERVOIR_K * sizeof(Reservoir)));
    }
    
    ~GpuContext() {
        if (reservoir_state) cudaFree(reservoir_state);
        if (diff_map) cudaFree(diff_map);
        if (morph_temp) cudaFree(morph_temp);
        if (morph_dest) cudaFree(morph_dest);
        if (final_mask) cudaFree(final_mask);
        if (rand_states) cudaFree(rand_states);
    }
};

static GpuContext g_gpu_ctx;

// Device helper functions
__device__ inline int color_diff(const rgb8& a, const rgb8& b) {
    return abs(int(a.r) - int(b.r)) + 
           abs(int(a.g) - int(b.g)) + 
           abs(int(a.b) - int(b.b));
}

__device__ inline bool is_inside(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

// Initialize random states
__global__ void init_rand_kernel(curandState* states, int width, int height, unsigned long seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Reservoir update and difference computation kernel
__global__ void update_reservoir_kernel(
    ImageView<rgb8> in,
    Reservoir* reservoir_state,
    uint8_t* diff_map,
    curandState* rand_states,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int p_idx = y * width + x;
    rgb8* line_ptr = (rgb8*)((char*)in.buffer + y * in.stride);
    rgb8 current_pixel = line_ptr[x];
    
    int base_idx = p_idx * RESERVOIR_K;
    int best_match = -1;
    
    // Search for matching reservoir
    for (int k = 0; k < RESERVOIR_K; ++k) {
        Reservoir& r = reservoir_state[base_idx + k];
        if (r.weight > 0 && color_diff(current_pixel, r.color) < RGB_DIFF_THRESHOLD) {
            best_match = k;
            break;
        }
    }
    
    if (best_match != -1) {
        // Update existing reservoir with moving average
        Reservoir& r = reservoir_state[base_idx + best_match];
        r.color.r = (uint8_t)(((r.weight - 1.0f) * r.color.r + current_pixel.r) / r.weight);
        r.color.g = (uint8_t)(((r.weight - 1.0f) * r.color.g + current_pixel.g) / r.weight);
        r.color.b = (uint8_t)(((r.weight - 1.0f) * r.color.b + current_pixel.b) / r.weight);
        if (r.weight < MAX_WEIGHT) r.weight++;
    } else {
        // No match found - try to find empty slot
        int empty_slot = -1;
        for (int k = 0; k < RESERVOIR_K; ++k) {
            if (reservoir_state[base_idx + k].weight == 0) {
                empty_slot = k;
                break;
            }
        }
        
        if (empty_slot != -1) {
            // Insert into empty slot
            reservoir_state[base_idx + empty_slot].color = current_pixel;
            reservoir_state[base_idx + empty_slot].weight = 1;
        } else {
            // Weighted reservoir sampling - find minimum weight
            int min_w_idx = 0;
            int min_w = reservoir_state[base_idx].weight;
            int sum_w = 0;
            
            for (int k = 0; k < RESERVOIR_K; ++k) {
                sum_w += reservoir_state[base_idx + k].weight;
                if (reservoir_state[base_idx + k].weight < min_w) {
                    min_w = reservoir_state[base_idx + k].weight;
                    min_w_idx = k;
                }
            }
            
            float rand_val = curand_uniform(&rand_states[p_idx]);
            if (rand_val * sum_w < reservoir_state[base_idx + min_w_idx].weight) {
                reservoir_state[base_idx + min_w_idx].color = current_pixel;
                reservoir_state[base_idx + min_w_idx].weight = 1;
            }
        }
    }
    
    // Compute difference from most stable background
    rgb8 bg_color = {0, 0, 0};
    int max_w = -1;
    for (int k = 0; k < RESERVOIR_K; ++k) {
        if (reservoir_state[base_idx + k].weight > max_w) {
            max_w = reservoir_state[base_idx + k].weight;
            bg_color = reservoir_state[base_idx + k].color;
        }
    }
    
    int diff = color_diff(current_pixel, bg_color) / 3;
    diff_map[p_idx] = (uint8_t)min(255, diff);
}

// Morphological erosion kernel
__global__ void erosion_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint8_t min_val = 255;
    for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
        for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
            if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (is_inside(nx, ny, width, height)) {
                min_val = min(min_val, input[ny * width + nx]);
            }
        }
    }
    output[y * width + x] = min_val;
}

// Morphological dilation kernel
__global__ void dilation_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    uint8_t max_val = 0;
    for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
        for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
            if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (is_inside(nx, ny, width, height)) {
                max_val = max(max_val, input[ny * width + nx]);
            }
        }
    }
    output[y * width + x] = max_val;
}

// Initialize mask with high threshold pixels
__global__ void init_hysteresis_kernel(
    const uint8_t* morph_dest,
    uint8_t* final_mask,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    final_mask[idx] = (morph_dest[idx] >= HYST_HIGH) ? 1 : 0;
}

// Hysteresis propagation kernel (iterative)
__global__ void hysteresis_propagate_kernel(
    const uint8_t* morph_dest,
    uint8_t* final_mask,
    int* changed,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // If already marked, check neighbors
    if (final_mask[idx] == 1) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = x + dx;
                int ny = y + dy;
                
                if (is_inside(nx, ny, width, height)) {
                    int nidx = ny * width + nx;
                    if (final_mask[nidx] == 0 && morph_dest[nidx] >= HYST_LOW) {
                        final_mask[nidx] = 1;
                        *changed = 1;
                    }
                }
            }
        }
    }
}

// Apply red overlay to detected regions
__global__ void apply_overlay_kernel(
    ImageView<rgb8> in,
    const uint8_t* final_mask,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    rgb8* line_ptr = (rgb8*)((char*)in.buffer + y * in.stride);
    
    if (final_mask[y * width + x]) {
        line_ptr[x].r = min(255, line_ptr[x].r + 100);
    }
}

void compute_cu(ImageView<rgb8> in)
{
    static bool initialized = false;
    
    // Print GPU info on first run
    if (!initialized) {
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        printf("=== CUDA GPU Mode ===\n");
        printf("Using GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Image size: %dx%d\n", in.width, in.height);
    }
    
    g_gpu_ctx.resize(in.width, in.height);
    
    dim3 block(16, 16);
    dim3 grid((in.width + block.x - 1) / block.x, (in.height + block.y - 1) / block.y);
    
    // Initialize random states on first run
    if (!initialized) {
        init_rand_kernel<<<grid, block>>>(g_gpu_ctx.rand_states, in.width, in.height, time(NULL));
        CUDA_CHECK_KERNEL();
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("GPU kernels initialized successfully\n");
        initialized = true;
    }
    
    // Copy input image to device
    Image<rgb8> device_in(in.width, in.height, true);
    CUDA_CHECK(cudaMemcpy2D(device_in.buffer, device_in.stride, in.buffer, in.stride, 
                 in.width * sizeof(rgb8), in.height, cudaMemcpyHostToDevice));
    
    // 1. Update reservoirs and compute difference map
    update_reservoir_kernel<<<grid, block>>>(
        device_in, g_gpu_ctx.reservoir_state, g_gpu_ctx.diff_map,
        g_gpu_ctx.rand_states, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 2. Morphological operations: erosion
    erosion_kernel<<<grid, block>>>(
        g_gpu_ctx.diff_map, g_gpu_ctx.morph_temp, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 3. Morphological operations: dilation
    dilation_kernel<<<grid, block>>>(
        g_gpu_ctx.morph_temp, g_gpu_ctx.morph_dest, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 4. Initialize hysteresis mask
    init_hysteresis_kernel<<<grid, block>>>(
        g_gpu_ctx.morph_dest, g_gpu_ctx.final_mask, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 5. Hysteresis propagation (iterative)
    int* d_changed;
    int h_changed = 1;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    
    // Iterate until no changes
    int max_iterations = 100;
    for (int iter = 0; iter < max_iterations && h_changed; ++iter) {
        h_changed = 0;
        CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        hysteresis_propagate_kernel<<<grid, block>>>(
            g_gpu_ctx.morph_dest, g_gpu_ctx.final_mask, d_changed, in.width, in.height
        );
        CUDA_CHECK_KERNEL();
        
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_changed));
    
    // 6. Apply red overlay to input image
    apply_overlay_kernel<<<grid, block>>>(
        device_in, g_gpu_ctx.final_mask, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy2D(in.buffer, in.stride, device_in.buffer, device_in.stride, 
                 in.width * sizeof(rgb8), in.height, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaDeviceSynchronize());
}