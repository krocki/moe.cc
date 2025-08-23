/**
 * Memory-mapped BinFile loading interface
 * 
 * Provides the same BinFile interface as io.h but uses memory mapping
 * for efficient access to large model files.
 */

#ifndef IO_MMAP_H
#define IO_MMAP_H

#include "io.h"

/**
 * Load binary file using memory mapping
 * Compatible interface with bin_load() from io.h
 * 
 * @param path: Path to binary model file
 * @return: BinFile* compatible with existing code, or NULL on error
 */
BinFile* bin_load_mmap(const char* path);

/**
 * Free memory-mapped BinFile and clean up resources
 * Use this instead of bin_free() for mmap'd files
 * 
 * @param bf: BinFile loaded with bin_load_mmap()
 */
void bin_free_mmap(BinFile* bf);

/**
 * Alias for bin_load_mmap() to enable drop-in replacement
 */
BinFile* bin_load_mmap_interface(const char* path);

#endif // IO_MMAP_H