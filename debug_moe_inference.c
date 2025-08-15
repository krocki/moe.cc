#include "model.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.bin>\n", argv[1]);
        return 1;
    }
    
    BinFile* bf = bin_load(argv[1]);
    if (!bf) {
        fprintf(stderr, "Failed to load %s\n", argv[1]);
        return 1;
    }
    
    fprintf(stderr, "Debug: Starting MoE inference...\n");
    
    int E = 0, d_ff = -1;
    for (;;) {
        char k_down[256];
        snprintf(k_down, sizeof(k_down), "model.layers.0.mlp.experts.%d.down_proj.weight", E);
        TensorBin* t = bin_find(bf, k_down);
        fprintf(stderr, "Debug: Looking for %s -> %s\n", k_down, t ? "FOUND" : "NOT FOUND");
        
        // If FP32 version not found, try quantized versions
        if (!t) {
            snprintf(k_down, sizeof(k_down), "model.layers.0.mlp.experts.%d.down_proj.weight.q8", E);
            t = bin_find(bf, k_down);
            fprintf(stderr, "Debug: Looking for %s -> %s\n", k_down, t ? "FOUND" : "NOT FOUND");
        }
        if (!t) {
            snprintf(k_down, sizeof(k_down), "model.layers.0.mlp.experts.%d.down_proj.weight.q4", E);
            t = bin_find(bf, k_down);
            fprintf(stderr, "Debug: Looking for %s -> %s\n", k_down, t ? "FOUND" : "NOT FOUND");
        }
        
        if (!t) {
            fprintf(stderr, "Debug: No more experts found. Total: %d\n", E);
            break;
        }
        
        d_ff = t->shape[1];
        // For Q4, shape[1] is packed (2 values per byte), so actual cols is shape[1] * 2
        if (strstr(k_down, ".q4")) d_ff *= 2;
        
        fprintf(stderr, "Debug: Expert %d found, d_ff = %d (from tensor: %s)\n", E, d_ff, k_down);
        E++;
        
        if (E >= 5) break; // Just check first 5
    }
    
    fprintf(stderr, "Debug: Final result: E=%d, d_ff=%d\n", E, d_ff);
    
    bin_free(bf);
    return 0;
}