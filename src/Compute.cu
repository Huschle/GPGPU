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

// Structure of Arrays (SoA) layout for better memory coalescing
struct ReservoirSoA {
    uint8_t* r_colors;
    uint8_t* g_colors;
    uint8_t* b_colors;
    uint16_t* weights;
};

struct GpuContext {
    int width = 0;
    int height = 0;
    Reservoir* reservoir_state = nullptr;
    
    // SOA buffers for reservoir access
    ReservoirSoA reservoir_soa;
    
    uint8_t* diff_map = nullptr;
    uint8_t* morph_temp = nullptr;
    uint8_t* morph_dest = nullptr;
    uint8_t* final_mask = nullptr;
    curandState* rand_states = nullptr;
    
    void resize(int w, int h) {
        if (width == w && height == h) return;
        
        // Free old buffers
        if (reservoir_state) CUDA_CHECK(cudaFree(reservoir_state));
        if (reservoir_soa.r_colors) CUDA_CHECK(cudaFree(reservoir_soa.r_colors));
        if (reservoir_soa.g_colors) CUDA_CHECK(cudaFree(reservoir_soa.g_colors));
        if (reservoir_soa.b_colors) CUDA_CHECK(cudaFree(reservoir_soa.b_colors));
        if (reservoir_soa.weights) CUDA_CHECK(cudaFree(reservoir_soa.weights));
        if (diff_map) CUDA_CHECK(cudaFree(diff_map));
        if (morph_temp) CUDA_CHECK(cudaFree(morph_temp));
        if (morph_dest) CUDA_CHECK(cudaFree(morph_dest));
        if (final_mask) CUDA_CHECK(cudaFree(final_mask));
        if (rand_states) CUDA_CHECK(cudaFree(rand_states));
        
        width = w;
        height = h;
        
        // Allocate SoA buffers for better memory coalescing
        size_t total_reservoirs = w * h * RESERVOIR_K;
        CUDA_CHECK(cudaMalloc(&reservoir_soa.r_colors, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&reservoir_soa.g_colors, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&reservoir_soa.b_colors, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&reservoir_soa.weights, total_reservoirs * sizeof(uint16_t)));
        
        // Legacy AoS buffer (keep for compatibility during transition)
        CUDA_CHECK(cudaMalloc(&reservoir_state, w * h * RESERVOIR_K * sizeof(Reservoir)));
        
        CUDA_CHECK(cudaMalloc(&diff_map, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&morph_temp, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&morph_dest, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&final_mask, w * h * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&rand_states, w * h * sizeof(curandState)));
        
        // Initialize to zero
        CUDA_CHECK(cudaMemset(reservoir_state, 0, w * h * RESERVOIR_K * sizeof(Reservoir)));
        CUDA_CHECK(cudaMemset(reservoir_soa.r_colors, 0, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(reservoir_soa.g_colors, 0, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(reservoir_soa.b_colors, 0, total_reservoirs * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(reservoir_soa.weights, 0, total_reservoirs * sizeof(uint16_t)));
    }
    
    ~GpuContext() {
        if (reservoir_state) cudaFree(reservoir_state);
        if (reservoir_soa.r_colors) cudaFree(reservoir_soa.r_colors);
        if (reservoir_soa.g_colors) cudaFree(reservoir_soa.g_colors);
        if (reservoir_soa.b_colors) cudaFree(reservoir_soa.b_colors);
        if (reservoir_soa.weights) cudaFree(reservoir_soa.weights);
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

// Optimized reservoir update with SoA layout for better memory coalescing
__global__ void update_reservoir_soa_kernel(
    ImageView<rgb8> in,
    uint8_t* r_colors,
    uint8_t* g_colors,
    uint8_t* b_colors,
    uint16_t* weights,
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
    
    // SoA: Each reservoir channel stored contiguously
    // Layout: [pixel0_res0, pixel1_res0, ..., pixelN_res0, pixel0_res1, ...]
    int base_idx = p_idx * RESERVOIR_K;
    int best_match = -1;
    
    // Search for matching reservoir - coalesced reads
    for (int k = 0; k < RESERVOIR_K; ++k) {
        int idx = base_idx + k;
        uint16_t w = weights[idx];
        if (w > 0) {
            uint8_t r = r_colors[idx];
            uint8_t g = g_colors[idx];
            uint8_t b = b_colors[idx];
            int diff = abs(int(current_pixel.r) - int(r)) +
                      abs(int(current_pixel.g) - int(g)) +
                      abs(int(current_pixel.b) - int(b));
            if (diff < RGB_DIFF_THRESHOLD) {
                best_match = k;
                break;
            }
        }
    }
    
    if (best_match != -1) {
        // Update existing reservoir with moving average - coalesced writes
        int idx = base_idx + best_match;
        uint16_t w = weights[idx];
        r_colors[idx] = (uint8_t)(((w - 1.0f) * r_colors[idx] + current_pixel.r) / w);
        g_colors[idx] = (uint8_t)(((w - 1.0f) * g_colors[idx] + current_pixel.g) / w);
        b_colors[idx] = (uint8_t)(((w - 1.0f) * b_colors[idx] + current_pixel.b) / w);
        if (w < MAX_WEIGHT) weights[idx] = w + 1;
    } else {
        // No match found - try to find empty slot
        int empty_slot = -1;
        for (int k = 0; k < RESERVOIR_K; ++k) {
            if (weights[base_idx + k] == 0) {
                empty_slot = k;
                break;
            }
        }
        
        if (empty_slot != -1) {
            // Insert into empty slot - coalesced writes
            int idx = base_idx + empty_slot;
            r_colors[idx] = current_pixel.r;
            g_colors[idx] = current_pixel.g;
            b_colors[idx] = current_pixel.b;
            weights[idx] = 1;
        } else {
            // Weighted reservoir sampling - find minimum weight
            int min_w_idx = 0;
            uint16_t min_w = weights[base_idx];
            int sum_w = 0;
            
            for (int k = 0; k < RESERVOIR_K; ++k) {
                uint16_t w = weights[base_idx + k];
                sum_w += w;
                if (w < min_w) {
                    min_w = w;
                    min_w_idx = k;
                }
            }
            
            float rand_val = curand_uniform(&rand_states[p_idx]);
            if (rand_val * sum_w < min_w) {
                int idx = base_idx + min_w_idx;
                r_colors[idx] = current_pixel.r;
                g_colors[idx] = current_pixel.g;
                b_colors[idx] = current_pixel.b;
                weights[idx] = 1;
            }
        }
    }
    
    // Compute difference from most stable background - coalesced reads
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
    uint16_t max_w = 0;
    for (int k = 0; k < RESERVOIR_K; ++k) {
        int idx = base_idx + k;
        uint16_t w = weights[idx];
        if (w > max_w) {
            max_w = w;
            bg_r = r_colors[idx];
            bg_g = g_colors[idx];
            bg_b = b_colors[idx];
        }
    }
    
    int diff = (abs(int(current_pixel.r) - int(bg_r)) +
               abs(int(current_pixel.g) - int(bg_g)) +
               abs(int(current_pixel.b) - int(bg_b))) / 3;
    diff_map[p_idx] = (uint8_t)min(255, diff);
}

// Legacy AoS version kept for compatibility
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

// Morphological erosion kernel with shared memory optimization
__global__ void erosion_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    // Shared memory tile with halo for neighbor access
    // Block is 16x16, but we need halo of MORPH_RADIUS (3) on each side
    __shared__ uint8_t tile[16 + 2*MORPH_RADIUS][16 + 2*MORPH_RADIUS];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x + MORPH_RADIUS;
    int ty = threadIdx.y + MORPH_RADIUS;
    
    // Load center tile
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    } else {
        tile[ty][tx] = 255; // Padding value for erosion
    }
    
    // Load halo regions (left/right/top/bottom)
    if (threadIdx.x < MORPH_RADIUS) {
        // Left halo
        int halo_x = x - MORPH_RADIUS;
        if (halo_x >= 0 && y < height) {
            tile[ty][threadIdx.x] = input[y * width + halo_x];
        } else {
            tile[ty][threadIdx.x] = 255;
        }
        // Right halo
        halo_x = x + blockDim.x;
        if (halo_x < width && y < height) {
            tile[ty][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[y * width + halo_x];
        } else {
            tile[ty][threadIdx.x + blockDim.x + MORPH_RADIUS] = 255;
        }
    }
    
    if (threadIdx.y < MORPH_RADIUS) {
        // Top halo
        int halo_y = y - MORPH_RADIUS;
        if (x < width && halo_y >= 0) {
            tile[threadIdx.y][tx] = input[halo_y * width + x];
        } else {
            tile[threadIdx.y][tx] = 255;
        }
        // Bottom halo
        halo_y = y + blockDim.y;
        if (x < width && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][tx] = input[halo_y * width + x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][tx] = 255;
        }
    }
    
    // Load corner halos
    if (threadIdx.x < MORPH_RADIUS && threadIdx.y < MORPH_RADIUS) {
        // Top-left
        int halo_x = x - MORPH_RADIUS;
        int halo_y = y - MORPH_RADIUS;
        if (halo_x >= 0 && halo_y >= 0) {
            tile[threadIdx.y][threadIdx.x] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y][threadIdx.x] = 255;
        }
        
        // Top-right
        halo_x = x + blockDim.x;
        if (halo_x < width && halo_y >= 0) {
            tile[threadIdx.y][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y][threadIdx.x + blockDim.x + MORPH_RADIUS] = 255;
        }
        
        // Bottom-left
        halo_x = x - MORPH_RADIUS;
        halo_y = y + blockDim.y;
        if (halo_x >= 0 && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x] = 255;
        }
        
        // Bottom-right
        halo_x = x + blockDim.x;
        if (halo_x < width && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x + blockDim.x + MORPH_RADIUS] = 255;
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    // Compute erosion using shared memory
    uint8_t min_val = 255;
    for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
        for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
            if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue;
            min_val = min(min_val, tile[ty + dy][tx + dx]);
        }
    }
    output[y * width + x] = min_val;
}

// Morphological dilation kernel with shared memory optimization
__global__ void dilation_kernel(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height
) {
    // Shared memory tile with halo
    __shared__ uint8_t tile[16 + 2*MORPH_RADIUS][16 + 2*MORPH_RADIUS];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x + MORPH_RADIUS;
    int ty = threadIdx.y + MORPH_RADIUS;
    
    // Load center tile
    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    } else {
        tile[ty][tx] = 0; // Padding value for dilation
    }
    
    // Load halo regions
    if (threadIdx.x < MORPH_RADIUS) {
        // Left halo
        int halo_x = x - MORPH_RADIUS;
        if (halo_x >= 0 && y < height) {
            tile[ty][threadIdx.x] = input[y * width + halo_x];
        } else {
            tile[ty][threadIdx.x] = 0;
        }
        // Right halo
        halo_x = x + blockDim.x;
        if (halo_x < width && y < height) {
            tile[ty][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[y * width + halo_x];
        } else {
            tile[ty][threadIdx.x + blockDim.x + MORPH_RADIUS] = 0;
        }
    }
    
    if (threadIdx.y < MORPH_RADIUS) {
        // Top halo
        int halo_y = y - MORPH_RADIUS;
        if (x < width && halo_y >= 0) {
            tile[threadIdx.y][tx] = input[halo_y * width + x];
        } else {
            tile[threadIdx.y][tx] = 0;
        }
        // Bottom halo
        halo_y = y + blockDim.y;
        if (x < width && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][tx] = input[halo_y * width + x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][tx] = 0;
        }
    }
    
    // Load corner halos
    if (threadIdx.x < MORPH_RADIUS && threadIdx.y < MORPH_RADIUS) {
        // Top-left
        int halo_x = x - MORPH_RADIUS;
        int halo_y = y - MORPH_RADIUS;
        if (halo_x >= 0 && halo_y >= 0) {
            tile[threadIdx.y][threadIdx.x] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Top-right
        halo_x = x + blockDim.x;
        if (halo_x < width && halo_y >= 0) {
            tile[threadIdx.y][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y][threadIdx.x + blockDim.x + MORPH_RADIUS] = 0;
        }
        
        // Bottom-left
        halo_x = x - MORPH_RADIUS;
        halo_y = y + blockDim.y;
        if (halo_x >= 0 && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x] = 0;
        }
        
        // Bottom-right
        halo_x = x + blockDim.x;
        if (halo_x < width && halo_y < height) {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x + blockDim.x + MORPH_RADIUS] = input[halo_y * width + halo_x];
        } else {
            tile[threadIdx.y + blockDim.y + MORPH_RADIUS][threadIdx.x + blockDim.x + MORPH_RADIUS] = 0;
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    // Compute dilation using shared memory
    uint8_t max_val = 0;
    for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
        for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
            if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue;
            max_val = max(max_val, tile[ty + dy][tx + dx]);
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

// Hysteresis propagation kernel with shared memory (iterative)
__global__ void hysteresis_propagate_kernel(
    const uint8_t* morph_dest,
    uint8_t* final_mask,
    int* changed,
    int width,
    int height
) {
    // Shared memory for mask tile with halo
    __shared__ uint8_t mask_tile[16 + 2][16 + 2];
    __shared__ uint8_t morph_tile[16 + 2][16 + 2];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Load center tile for both mask and morph_dest
    if (x < width && y < height) {
        int idx = y * width + x;
        mask_tile[ty][tx] = final_mask[idx];
        morph_tile[ty][tx] = morph_dest[idx];
    } else {
        mask_tile[ty][tx] = 0;
        morph_tile[ty][tx] = 0;
    }
    
    // Load halo (1 pixel border for 8-connectivity)
    // Left
    if (threadIdx.x == 0) {
        int halo_x = x - 1;
        if (halo_x >= 0 && y < height) {
            int idx = y * width + halo_x;
            mask_tile[ty][0] = final_mask[idx];
            morph_tile[ty][0] = morph_dest[idx];
        } else {
            mask_tile[ty][0] = 0;
            morph_tile[ty][0] = 0;
        }
    }
    // Right
    if (threadIdx.x == blockDim.x - 1) {
        int halo_x = x + 1;
        if (halo_x < width && y < height) {
            int idx = y * width + halo_x;
            mask_tile[ty][blockDim.x + 1] = final_mask[idx];
            morph_tile[ty][blockDim.x + 1] = morph_dest[idx];
        } else {
            mask_tile[ty][blockDim.x + 1] = 0;
            morph_tile[ty][blockDim.x + 1] = 0;
        }
    }
    // Top
    if (threadIdx.y == 0) {
        int halo_y = y - 1;
        if (x < width && halo_y >= 0) {
            int idx = halo_y * width + x;
            mask_tile[0][tx] = final_mask[idx];
            morph_tile[0][tx] = morph_dest[idx];
        } else {
            mask_tile[0][tx] = 0;
            morph_tile[0][tx] = 0;
        }
    }
    // Bottom
    if (threadIdx.y == blockDim.y - 1) {
        int halo_y = y + 1;
        if (x < width && halo_y < height) {
            int idx = halo_y * width + x;
            mask_tile[blockDim.y + 1][tx] = final_mask[idx];
            morph_tile[blockDim.y + 1][tx] = morph_dest[idx];
        } else {
            mask_tile[blockDim.y + 1][tx] = 0;
            morph_tile[blockDim.y + 1][tx] = 0;
        }
    }
    
    // Load corners
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        // Top-left
        int halo_x = x - 1, halo_y = y - 1;
        if (halo_x >= 0 && halo_y >= 0) {
            int idx = halo_y * width + halo_x;
            mask_tile[0][0] = final_mask[idx];
            morph_tile[0][0] = morph_dest[idx];
        } else {
            mask_tile[0][0] = 0;
            morph_tile[0][0] = 0;
        }
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        // Top-right
        int halo_x = x + 1, halo_y = y - 1;
        if (halo_x < width && halo_y >= 0) {
            int idx = halo_y * width + halo_x;
            mask_tile[0][blockDim.x + 1] = final_mask[idx];
            morph_tile[0][blockDim.x + 1] = morph_dest[idx];
        } else {
            mask_tile[0][blockDim.x + 1] = 0;
            morph_tile[0][blockDim.x + 1] = 0;
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        // Bottom-left
        int halo_x = x - 1, halo_y = y + 1;
        if (halo_x >= 0 && halo_y < height) {
            int idx = halo_y * width + halo_x;
            mask_tile[blockDim.y + 1][0] = final_mask[idx];
            morph_tile[blockDim.y + 1][0] = morph_dest[idx];
        } else {
            mask_tile[blockDim.y + 1][0] = 0;
            morph_tile[blockDim.y + 1][0] = 0;
        }
    }
    if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        // Bottom-right
        int halo_x = x + 1, halo_y = y + 1;
        if (halo_x < width && halo_y < height) {
            int idx = halo_y * width + halo_x;
            mask_tile[blockDim.y + 1][blockDim.x + 1] = final_mask[idx];
            morph_tile[blockDim.y + 1][blockDim.x + 1] = morph_dest[idx];
        } else {
            mask_tile[blockDim.y + 1][blockDim.x + 1] = 0;
            morph_tile[blockDim.y + 1][blockDim.x + 1] = 0;
        }
    }
    
    __syncthreads();
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // If any neighbor is marked, propagate to this pixel if it passes low threshold
    if (mask_tile[ty][tx] == 0 && morph_tile[ty][tx] >= HYST_LOW) {
        bool has_marked_neighbor = false;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                if (mask_tile[ty + dy][tx + dx] == 1) {
                    has_marked_neighbor = true;
                    break;
                }
            }
            if (has_marked_neighbor) break;
        }
        
        if (has_marked_neighbor) {
            final_mask[idx] = 1;
            atomicOr(changed, 1);
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
    static bool use_optimized = true; // Flag to switch between AoS and SoA
    
    // Print GPU info on first run
    if (!initialized) {
        int device;
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        printf("CUDA GPU Mode (with Shared Memory and SoA)\n");
        printf("Using GPU: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Image size: %dx%d\n", in.width, in.height);
        printf("Optimizations enabled:\n");
        printf("  - Shared memory for morphology (erosion/dilation)\n");
        printf("  - Shared memory for hysteresis propagation\n");
        printf("  - Structure of Arrays (SoA) for reservoir sampling\n");
        printf("  - Memory coalescing optimization\n");
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
    
    // 1. Update reservoirs and compute difference map (using SoA layout)
    if (use_optimized) {
        update_reservoir_soa_kernel<<<grid, block>>>(
            device_in,
            g_gpu_ctx.reservoir_soa.r_colors,
            g_gpu_ctx.reservoir_soa.g_colors,
            g_gpu_ctx.reservoir_soa.b_colors,
            g_gpu_ctx.reservoir_soa.weights,
            g_gpu_ctx.diff_map,
            g_gpu_ctx.rand_states,
            in.width, in.height
        );
    } else {
        // Fallback to legacy AoS version
        update_reservoir_kernel<<<grid, block>>>(
            device_in, g_gpu_ctx.reservoir_state, g_gpu_ctx.diff_map,
            g_gpu_ctx.rand_states, in.width, in.height
        );
    }
    CUDA_CHECK_KERNEL();
    
    // 2. Morphological operations: erosion (using shared memory)
    erosion_kernel<<<grid, block>>>(
        g_gpu_ctx.diff_map, g_gpu_ctx.morph_temp, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 3. Morphological operations: dilation (using shared memory)
    dilation_kernel<<<grid, block>>>(
        g_gpu_ctx.morph_temp, g_gpu_ctx.morph_dest, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 4. Initialize hysteresis mask
    init_hysteresis_kernel<<<grid, block>>>(
        g_gpu_ctx.morph_dest, g_gpu_ctx.final_mask, in.width, in.height
    );
    CUDA_CHECK_KERNEL();
    
    // 5. Hysteresis propagation (using shared memory, iterative)
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