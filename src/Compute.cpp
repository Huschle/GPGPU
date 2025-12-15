#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <queue>


#define RESERVOIR_K 3
#define RGB_DIFF_THRESHOLD 30
#define MAX_WEIGHT 50
#define MORPH_RADIUS 3
#define HYST_LOW 4
#define HYST_HIGH 30

struct rgb8_pixel {
    uint8_t r, g, b;
};


struct Reservoir {
    rgb8 color;
    uint16_t weight;
};


struct CpuContext {
    int width = 0;
    int height = 0;
    

    std::vector<Reservoir> reservoir_state; 
    
    // Buffers intermédiaires
    std::vector<uint8_t> diff_map;      // Différence brute
    std::vector<uint8_t> morph_temp;    // Après érosion
    std::vector<uint8_t> morph_dest;    // Après dilatation
    std::vector<bool>    final_mask;    // Masque final après hystérésis

    void resize(int w, int h) {
        if (width == w && height == h) return;
        width = w;
        height = h;
        reservoir_state.resize(w * h * RESERVOIR_K, {{0,0,0}, 0});
        diff_map.resize(w * h);
        morph_temp.resize(w * h);
        morph_dest.resize(w * h);
        final_mask.resize(w * h);
    }
};

// Instance globale (statique) pour la persistance entre les frames
static CpuContext g_ctx;

// Distance de Manhattan pour la couleur (plus rapide que Euclidienne)
inline int color_diff(const rgb8& a, const rgb8& b) {
    return std::abs(int(a.r) - int(b.r)) + 
           std::abs(int(a.g) - int(b.g)) + 
           std::abs(int(a.b) - int(b.b));
}

// Vérifie si un point est dans l'image
inline bool is_inside(int x, int y, int w, int h) {
    return x >= 0 && x < w && y >= 0 && y < h;
}

// --- Implémentation CPU ---

void compute_cpp(ImageView<rgb8> in)
{
    
    g_ctx.resize(in.width, in.height);
    
    int w = in.width;
    int h = in.height;

    // Pour chaque pixel de l'image
    for (int y = 0; y < h; ++y) {
        rgb8* line_ptr = (rgb8*)((char*)in.buffer + y * in.stride);
        
        for (int x = 0; x < w; ++x) {
            int p_idx = y * w + x;
            rgb8 current_pixel = line_ptr[x];
            
            
            int best_match = -1;
            int base_idx = p_idx * RESERVOIR_K; // Index dans le vecteur de réservoirs

            // Recherche d'un réservoir correspondant
            for (int k = 0; k < RESERVOIR_K; ++k) {
                Reservoir& r = g_ctx.reservoir_state[base_idx + k];
                if (r.weight > 0 && color_diff(current_pixel, r.color) < RGB_DIFF_THRESHOLD) {
                    best_match = k;
                    break;
                }
            }

            if (best_match != -1) {
                // mise à jour moyenne mobile
                Reservoir& r = g_ctx.reservoir_state[base_idx + best_match];
                r.color.r = (uint8_t)(((r.weight - 1.0f) * r.color.r + current_pixel.r) / r.weight);
                r.color.g = (uint8_t)(((r.weight - 1.0f) * r.color.g + current_pixel.g) / r.weight);
                r.color.b = (uint8_t)(((r.weight - 1.0f) * r.color.b + current_pixel.b) / r.weight);
                if (r.weight < MAX_WEIGHT) r.weight++;
            } else {
                // Pas de correspondance
                int empty_slot = -1;
                for (int k = 0; k < RESERVOIR_K; ++k) {
                    if (g_ctx.reservoir_state[base_idx + k].weight == 0) {
                        empty_slot = k;
                        break;
                    }
                }

                if (empty_slot != -1) {
                    // Insertion dans un slot vide
                    g_ctx.reservoir_state[base_idx + empty_slot] = {current_pixel, 1};
                } else {
                    // Remplacement aléatoire (Weighted Reservoir Sampling)
                    // on remplace celui qui a le poids min si proba ok
                    int min_w_idx = 0;
                    int min_w = g_ctx.reservoir_state[base_idx].weight;
                    int sum_w = 0;
                    
                    for(int k=0; k<RESERVOIR_K; ++k) {
                        sum_w += g_ctx.reservoir_state[base_idx+k].weight;
                        if(g_ctx.reservoir_state[base_idx+k].weight < min_w) {
                            min_w = g_ctx.reservoir_state[base_idx+k].weight;
                            min_w_idx = k;
                        }
                    }
                    
                    float rand_val = (float)rand() / RAND_MAX;
                    if (rand_val * sum_w < g_ctx.reservoir_state[base_idx + min_w_idx].weight) {
                         g_ctx.reservoir_state[base_idx + min_w_idx] = {current_pixel, 1};
                    }
                }
            }

            rgb8 bg_color = {0, 0, 0};
            int max_w = -1;
            for (int k = 0; k < RESERVOIR_K; ++k) {
                if (g_ctx.reservoir_state[base_idx + k].weight > max_w) {
                    max_w = g_ctx.reservoir_state[base_idx + k].weight;
                    bg_color = g_ctx.reservoir_state[base_idx + k].color;
                }
            }

            int diff = color_diff(current_pixel, bg_color) / 3;
            g_ctx.diff_map[p_idx] = (uint8_t)std::min(255, diff);
        }
    }
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t min_val = 255;
            for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
                for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
                    if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue; // Masque circulaire
                    int nx = x + dx;
                    int ny = y + dy;
                    if (is_inside(nx, ny, w, h)) {
                        min_val = std::min(min_val, g_ctx.diff_map[ny * w + nx]);
                    }
                }
            }
            g_ctx.morph_temp[y * w + x] = min_val;
        }
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t max_val = 0;
            for (int dy = -MORPH_RADIUS; dy <= MORPH_RADIUS; ++dy) {
                for (int dx = -MORPH_RADIUS; dx <= MORPH_RADIUS; ++dx) {
                    if (dx*dx + dy*dy > MORPH_RADIUS*MORPH_RADIUS) continue;
                    int nx = x + dx;
                    int ny = y + dy;
                    if (is_inside(nx, ny, w, h)) {
                        max_val = std::max(max_val, g_ctx.morph_temp[ny * w + nx]);
                    }
                }
            }
            g_ctx.morph_dest[y * w + x] = max_val;
            g_ctx.final_mask[y * w + x] = false;
        }
    }

    std::queue<std::pair<int, int>> q;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (g_ctx.morph_dest[y * w + x] >= HYST_HIGH) {
                q.push({x, y});
                g_ctx.final_mask[y * w + x] = true;
            }
        }
    }

    while(!q.empty()) {
        auto p = q.front();
        q.pop();
        int cx = p.first;
        int cy = p.second;

        // 8-connexité
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = cx + dx;
                int ny = cy + dy;
                
                if (is_inside(nx, ny, w, h)) {
                    int idx = ny * w + nx;
                    if (!g_ctx.final_mask[idx] && g_ctx.morph_dest[idx] >= HYST_LOW) {
                        g_ctx.final_mask[idx] = true;
                        q.push({nx, ny});
                    }
                }
            }
        }
    }

    // Modifier l'image d'entrée (input + rouge si masque actif)
    for (int y = 0; y < h; ++y) {
        rgb8* line_ptr = (rgb8*)((char*)in.buffer + y * in.stride);
        for (int x = 0; x < w; ++x) {
            if (g_ctx.final_mask[y * w + x]) {
                // On sature le rouge et on garde un peu du reste
                line_ptr[x].r = std::min(255, line_ptr[x].r + 100); 
                // supprimer G et B pour faire rouge pur comme la démo
                // line_ptr[x].g /= 2;
                // line_ptr[x].b /= 2;
            }
        }
    }
}


/// Your CUDA version of the algorithm
void compute_cu(ImageView<rgb8> in); // Déclaration

extern "C" {

  static Parameters g_params;

  void cpt_init(Parameters* params)
  {
    g_params = *params;
    // Initialisation du random seed
    srand(time(NULL));
  }

  void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
  {
    auto img = ImageView<rgb8>{(rgb8*)buffer, width, height, stride};
    if (g_params.device == e_device_t::CPU)
      compute_cpp(img);
    else if (g_params.device == e_device_t::GPU)
      compute_cu(img);
  }

}