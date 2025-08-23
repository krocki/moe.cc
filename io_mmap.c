/**
 * Memory-mapped implementation of the BinFile loading interface
 * 
 * This file provides an alternative implementation of bin_load() that uses
 * memory-mapped files instead of explicit loading into allocated memory.
 * The interface remains identical to the original io.c implementation.
 * 
 * Benefits of mmap approach:
 * - Lower memory usage (OS can page out unused parts)
 * - Faster startup (no need to read entire file)
 * - Shared memory pages between processes using same model
 * 
 * Usage:
 * - Replace bin_load() calls with bin_load_mmap()
 * - Use same bin_free() and bin_find() functions
 * - Tensor data pointers point directly into mapped memory
 */

#define _POSIX_C_SOURCE 200809L
#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * Extended BinFile structure for mmap support
 * Contains the original BinFile fields plus mmap-specific metadata
 */
typedef struct {
    BinFile base;           // Standard BinFile interface
    void* mmap_ptr;         // Pointer to memory-mapped region
    size_t mmap_size;       // Size of mapped region
    int fd;                 // File descriptor (kept open for mmap)
    char* filepath;         // Path to file (for error reporting)
} BinFileMmap;

/**
 * Read 32-bit unsigned integer from memory buffer
 * Handles proper byte order and bounds checking
 */
static inline uint32_t read_u32_from_memory(const uint8_t** ptr, const uint8_t* end) {
    if (*ptr + 4 > end) {
        fprintf(stderr, "Unexpected end of file while reading u32\n");
        exit(1);
    }
    uint32_t value;
    memcpy(&value, *ptr, 4);
    *ptr += 4;
    return value;
}

/**
 * Memory-mapped implementation of bin_load()
 * 
 * This function provides the same interface as the original bin_load() but uses
 * memory mapping for efficient access to large model files.
 * 
 * @param path: Path to binary model file  
 * @return: BinFile* compatible with existing code, or NULL on error
 */
BinFile* bin_load_mmap(const char* path) {
    // Open file for reading
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return NULL;
    }
    
    // Get file size
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("fstat");
        close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;
    
    // Memory map the entire file
    void* mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return NULL;
    }
    
    // Parse the memory-mapped file
    const uint8_t* data = (const uint8_t*)mapped_data;
    const uint8_t* ptr = data;
    const uint8_t* end = data + file_size;
    
    // Validate file header (magic bytes)
    if (ptr + 6 > end) {
        fprintf(stderr, "File too small: %s\n", path);
        goto error_cleanup;
    }
    
    const uint8_t ref_v1[6] = {'Q','W','3','W',0x00,0x01};
    const uint8_t ref_v2[6] = {'Q','W','3','W',0x00,0x02};
    int file_version = 1;
    
    if (memcmp(ptr, ref_v1, 6) == 0) {
        file_version = 1;
    } else if (memcmp(ptr, ref_v2, 6) == 0) {
        file_version = 2;
    } else {
        fprintf(stderr, "Invalid magic bytes in %s\n", path);
        goto error_cleanup;
    }
    ptr += 6;
    
    // Read tensor count from header
    uint32_t header_tensor_count = read_u32_from_memory(&ptr, end);
    
    // Allocate extended BinFile structure
    BinFileMmap* bf_mmap = (BinFileMmap*)calloc(1, sizeof(BinFileMmap));
    if (!bf_mmap) {
        fprintf(stderr, "Failed to allocate BinFileMmap\n");
        goto error_cleanup;
    }
    
    // Initialize mmap-specific fields
    bf_mmap->mmap_ptr = mapped_data;
    bf_mmap->mmap_size = file_size;
    bf_mmap->fd = fd;
    bf_mmap->filepath = strdup(path);
    
    // Initialize BinFile fields
    BinFile* bf = &bf_mmap->base;
    bf->count = 0;
    
    // Allocate tensor array with extra capacity for quantized tensors
    // (quantized models have .q8/.q4 + .scale pairs, roughly doubling count)
    size_t capacity = header_tensor_count * 2;
    bf->arr = (TensorBin*)calloc(capacity, sizeof(TensorBin));
    if (!bf->arr) {
        fprintf(stderr, "Failed to allocate tensor array\n");
        goto error_cleanup_bf;
    }
    
    // Parse tensor metadata and set up data pointers
    while (ptr < end && bf->count < capacity) {
        TensorBin* tensor = &bf->arr[bf->count];
        
        // Read tensor name
        if (ptr + 4 > end) break;
        uint32_t name_len = read_u32_from_memory(&ptr, end);
        
        if (ptr + name_len > end) {
            fprintf(stderr, "Invalid name length in tensor %d\n", bf->count);
            break;
        }
        
        tensor->name = (char*)malloc(name_len + 1);
        if (!tensor->name) {
            fprintf(stderr, "Failed to allocate tensor name\n");
            break;
        }
        memcpy(tensor->name, ptr, name_len);
        tensor->name[name_len] = 0;
        ptr += name_len;
        
        // Read tensor metadata
        tensor->dtype = (int)read_u32_from_memory(&ptr, end);
        tensor->ndim = (int)read_u32_from_memory(&ptr, end);
        
        // Read tensor shape
        tensor->shape = (int*)malloc(tensor->ndim * sizeof(int));
        if (!tensor->shape) {
            fprintf(stderr, "Failed to allocate tensor shape\n");
            free(tensor->name);
            break;
        }
        
        size_t total_elements = 1;
        for (int i = 0; i < tensor->ndim; i++) {
            tensor->shape[i] = (int)read_u32_from_memory(&ptr, end);
            total_elements *= (size_t)tensor->shape[i];
        }
        
        // Calculate tensor data size
        size_t bytes_per_element;
        switch (tensor->dtype) {
            case 0: bytes_per_element = 4; break;  // f32
            case 1: bytes_per_element = 2; break;  // f16
            case 2: bytes_per_element = 1; break;  // i8
            case 3: bytes_per_element = 1; break;  // i4 (packed)
            default:
                fprintf(stderr, "Unknown dtype %d in tensor %s\n", tensor->dtype, tensor->name);
                free(tensor->name);
                free(tensor->shape);
                goto error_cleanup_tensors;
        }
        tensor->nbytes = total_elements * bytes_per_element;
        
        // Set data pointer directly into memory-mapped region
        // This is the key difference from explicit loading - no data copying
        if (ptr + tensor->nbytes > end) {
            fprintf(stderr, "Tensor data extends beyond file end: %s\n", tensor->name);
            free(tensor->name);
            free(tensor->shape);
            break;
        }
        tensor->data = (void*)ptr;  // Point directly into mmap'd memory
        ptr += tensor->nbytes;
        
        // Read group_size for v2 format AFTER tensor data (matches original order)
        if (file_version >= 2) {
            tensor->group_size = (size_t)read_u32_from_memory(&ptr, end);
        } else {
            tensor->group_size = 0;
        }
        
        bf->count++;
        
        // Expand capacity if needed
        if (bf->count >= capacity) {
            capacity *= 2;
            TensorBin* new_arr = (TensorBin*)realloc(bf->arr, capacity * sizeof(TensorBin));
            if (!new_arr) {
                fprintf(stderr, "Failed to expand tensor array\n");
                break;
            }
            bf->arr = new_arr;
        }
    }
    
    return bf;
    
error_cleanup_tensors:
    // Clean up any partially loaded tensors
    for (int i = 0; i < bf->count; i++) {
        free(bf->arr[i].name);
        free(bf->arr[i].shape);
        // Note: don't free data pointers - they point into mmap'd memory
    }
    free(bf->arr);
    
error_cleanup_bf:
    free(bf_mmap->filepath);
    free(bf_mmap);
    
error_cleanup:
    munmap(mapped_data, file_size);
    close(fd);
    return NULL;
}

/**
 * Free memory-mapped BinFile and clean up resources
 * Compatible with original bin_free() but handles mmap cleanup
 */
void bin_free_mmap(BinFile* bf) {
    if (!bf) return;
    
    // Cast to extended structure to access mmap fields
    BinFileMmap* bf_mmap = (BinFileMmap*)bf;
    
    // Free tensor metadata (names and shapes)
    for (int i = 0; i < bf->count; i++) {
        free(bf->arr[i].name);
        free(bf->arr[i].shape);
        // Note: don't free data pointers - they point into mmap'd memory
    }
    free(bf->arr);
    
    // Clean up memory mapping
    if (bf_mmap->mmap_ptr) {
        if (munmap(bf_mmap->mmap_ptr, bf_mmap->mmap_size) != 0) {
            perror("munmap");
        }
    }
    
    // Close file descriptor
    if (bf_mmap->fd >= 0) {
        close(bf_mmap->fd);
    }
    
    // Free filepath string
    free(bf_mmap->filepath);
    
    // Free the structure itself
    free(bf_mmap);
}

/**
 * Alias for bin_load_mmap() to match original interface
 * This allows drop-in replacement of bin_load() calls
 */
BinFile* bin_load_mmap_interface(const char* path) {
    return bin_load_mmap(path);
}