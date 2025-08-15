#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file tokenizer.h
 * @brief Qwen3-30B-A3B Tokenizer - Compact C implementation for BPE tokenization
 * 
 * This tokenizer implements a BytePair Encoding (BPE) tokenizer compatible with
 * Qwen3-30B-A3B model. It supports:
 * - Token encoding: text -> token IDs
 * - Token decoding: token IDs -> text tokens  
 * - Chat template wrapping for conversational AI
 * - Binary serialization format for fast loading
 * 
 * The implementation uses:
 * - Hash map for token lookup (O(1) access)
 * - Trie structure for efficient prefix matching during encoding
 * - GPT-2 style byte encoding to handle arbitrary Unicode text
 * - Special token handling for chat formatting
 */

/**
 * @brief Maximum length of any single token in bytes
 */
#define MAX_TOKEN_LENGTH 256

/**
 * @brief Maximum number of tokens that can be stored
 */
#define MAX_VOCAB_SIZE 200000

/**
 * @brief Maximum length for chat templates
 */
#define MAX_CHAT_TEMPLATE_LENGTH 1024

/**
 * @brief Magic number for tokenizer binary format: "QW3T" (Qwen3 Tokenizer)
 */
#define TOKENIZER_MAGIC 0x54335751

/**
 * @brief Tokenizer binary format version
 */
#define TOKENIZER_VERSION 1

/**
 * @struct TokenizerResult
 * @brief Result structure for tokenizer operations
 * 
 * Contains token IDs and count for encoding operations, or
 * token strings and count for decoding operations.
 */
typedef struct {
    int* token_ids;          /** Array of token IDs (for encode result) */
    char** token_strings;    /** Array of token strings (for decode result) */
    int count;               /** Number of tokens */
    int capacity;            /** Allocated capacity */
} TokenizerResult;

/**
 * @struct Tokenizer
 * @brief Main tokenizer structure
 * 
 * Opaque structure containing:
 * - Vocabulary mapping (token string -> ID)
 * - Reverse mapping (ID -> token string)  
 * - Byte encoder/decoder for GPT-2 style byte encoding
 * - Merge rules for BPE algorithm
 * - Special tokens (BOS, EOS, etc.)
 * - Chat template for conversation formatting
 * - Trie structure for efficient token matching
 */
typedef struct Tokenizer Tokenizer;

/**
 * @brief Create a new tokenizer instance
 * @return Pointer to allocated Tokenizer structure, NULL on failure
 * 
 * Allocates memory for a new tokenizer. Must be freed with tokenizer_free().
 * The tokenizer is not functional until loaded with tokenizer_load().
 */
Tokenizer* tokenizer_create(void);

/**
 * @brief Free tokenizer memory
 * @param tokenizer Pointer to tokenizer instance
 * 
 * Deallocates all memory associated with the tokenizer including:
 * - Vocabulary tables
 * - Token strings
 * - Byte encoding tables
 * - Trie structure
 * - Chat template
 */
void tokenizer_free(Tokenizer* tokenizer);

/**
 * @brief Load tokenizer from binary file
 * @param tokenizer Pointer to tokenizer instance
 * @param filename Path to binary tokenizer file
 * @return 0 on success, -1 on failure
 * 
 * Loads tokenizer data from binary file created by export script.
 * Binary format contains:
 * - Header with magic number, version, vocab size
 * - Special token IDs (BOS, EOS, PAD)
 * - Chat template string
 * - Byte encoder mapping (256 entries)
 * - Vocabulary table (token ID -> token string)
 * - BPE merge rules
 * 
 * The file format is designed for fast loading and minimal memory overhead.
 */
int tokenizer_load(Tokenizer* tokenizer, const char* filename);

/**
 * @brief Encode text to token IDs
 * @param tokenizer Pointer to tokenizer instance
 * @param text Input text string (UTF-8 encoded)
 * @param result Pointer to TokenizerResult structure to store results
 * @return 0 on success, -1 on failure
 * 
 * Converts input text to sequence of token IDs using BPE algorithm:
 * 1. Split text on special tokens (e.g., <|im_start|>, <|im_end|>)
 * 2. Apply byte encoding to convert raw bytes to encoded Unicode
 * 3. Use greedy longest-match with trie structure to find tokens
 * 4. Apply BPE merge rules iteratively
 * 5. Return final sequence of token IDs
 * 
 * The result->token_ids array is allocated and must be freed by caller.
 * Use tokenizer_result_free() to properly clean up.
 */
int tokenizer_encode(Tokenizer* tokenizer, const char* text, TokenizerResult* result);

/**
 * @brief Decode token IDs to token strings
 * @param tokenizer Pointer to tokenizer instance  
 * @param token_ids Array of token IDs
 * @param num_tokens Number of token IDs in array
 * @param result Pointer to TokenizerResult structure to store results
 * @return 0 on success, -1 on failure
 * 
 * Converts sequence of token IDs back to token strings:
 * 1. Look up each token ID in vocabulary table
 * 2. For regular tokens: apply byte decoding to get original text
 * 3. For special tokens: return surface form directly
 * 4. Return array of token strings
 * 
 * The result->token_strings array and individual strings are allocated
 * and must be freed by caller. Use tokenizer_result_free().
 */
int tokenizer_decode(Tokenizer* tokenizer, const int* token_ids, int num_tokens, TokenizerResult* result);

/**
 * @brief Wrap message in chat template
 * @param tokenizer Pointer to tokenizer instance
 * @param message User message to wrap
 * @param result Buffer to store formatted chat string  
 * @param max_length Maximum length of result buffer
 * @return 0 on success, -1 on failure (buffer too small)
 * 
 * Formats user message using Qwen3 chat template:
 * <|im_start|>user
 * {message}<|im_end|>
 * <|im_start|>assistant
 * 
 * This prepares the text for model input in conversational format.
 * The assistant response will be generated after this prompt.
 */
int tokenizer_wrap_chat(Tokenizer* tokenizer, const char* message, char* result, int max_length);

/**
 * @brief Free TokenizerResult memory
 * @param result Pointer to TokenizerResult structure
 * 
 * Properly deallocates memory in TokenizerResult:
 * - For encode results: frees token_ids array
 * - For decode results: frees token_strings array and individual strings
 * - Resets count and capacity to 0
 */
void tokenizer_result_free(TokenizerResult* result);

/**
 * @brief Get vocabulary size
 * @param tokenizer Pointer to tokenizer instance
 * @return Number of tokens in vocabulary, -1 on error
 * 
 * Returns the total number of tokens in the tokenizer vocabulary.
 * This includes both regular BPE tokens and special tokens.
 */
int tokenizer_vocab_size(Tokenizer* tokenizer);

/**
 * @brief Get special token IDs
 * @param tokenizer Pointer to tokenizer instance
 * @param bos_id Pointer to store BOS (Beginning of Sequence) token ID
 * @param eos_id Pointer to store EOS (End of Sequence) token ID  
 * @param pad_id Pointer to store PAD token ID
 * @return 0 on success, -1 on failure
 * 
 * Retrieves commonly used special token IDs:
 * - BOS: Used to mark beginning of sequence (typically 151643)
 * - EOS: Used to mark end of sequence (typically 151645) 
 * - PAD: Used for padding in batched inference
 */
int tokenizer_get_special_tokens(Tokenizer* tokenizer, int* bos_id, int* eos_id, int* pad_id);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */