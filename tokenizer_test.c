#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @file tokenizer_test.c
 * @brief Comprehensive test suite for Qwen3 tokenizer implementation
 * 
 * This test suite validates all major functionality of the tokenizer:
 * - Loading from binary files
 * - Encoding text to token IDs
 * - Decoding token IDs back to text
 * - Chat template formatting
 * - Special token handling
 * - Edge cases and error conditions
 * 
 * Usage: ./tokenizer_test <tokenizer_binary_file>
 * 
 * The binary file should be generated using the Python export script:
 * python3 export_qwen3_tokenizer.py
 */

/* Test result counters */
static int tests_passed = 0;
static int tests_failed = 0;

/**
 * @brief Test assertion macro with detailed error reporting
 */
#define TEST_ASSERT(condition, test_name) \
    do { \
        if (condition) { \
            printf("✓ PASS: %s\n", test_name); \
            tests_passed++; \
        } else { \
            printf("✗ FAIL: %s\n", test_name); \
            tests_failed++; \
        } \
    } while (0)

/**
 * @brief Test tokenizer creation and destruction
 */
void test_tokenizer_lifecycle(void) {
    printf("\n=== Testing Tokenizer Lifecycle ===\n");
    
    /* Test creation */
    Tokenizer* tokenizer = tokenizer_create();
    TEST_ASSERT(tokenizer != NULL, "tokenizer_create() returns non-NULL");
    
    /* Test that vocab size is 0 before loading */
    TEST_ASSERT(tokenizer_vocab_size(tokenizer) == 0, "vocab_size is 0 before loading");
    
    /* Test freeing */
    tokenizer_free(tokenizer);
    TEST_ASSERT(1, "tokenizer_free() completes without crash");
    
    /* Test freeing NULL (should not crash) */
    tokenizer_free(NULL);
    TEST_ASSERT(1, "tokenizer_free(NULL) handles gracefully");
}

/**
 * @brief Test tokenizer loading from binary file
 */
void test_tokenizer_loading(const char* tokenizer_file) {
    printf("\n=== Testing Tokenizer Loading ===\n");
    
    Tokenizer* tokenizer = tokenizer_create();
    TEST_ASSERT(tokenizer != NULL, "tokenizer created for loading test");
    
    /* Test loading valid file */
    int result = tokenizer_load(tokenizer, tokenizer_file);
    TEST_ASSERT(result == 0, "tokenizer_load() succeeds");
    
    /* Test vocab size after loading */
    int vocab_size = tokenizer_vocab_size(tokenizer);
    TEST_ASSERT(vocab_size > 100000, "vocab_size is reasonable after loading");
    printf("  Loaded vocab size: %d\n", vocab_size);
    
    /* Test special tokens */
    int bos_id, eos_id, pad_id;
    result = tokenizer_get_special_tokens(tokenizer, &bos_id, &eos_id, &pad_id);
    TEST_ASSERT(result == 0, "tokenizer_get_special_tokens() succeeds");
    TEST_ASSERT(bos_id >= 0, "BOS token ID is valid");
    TEST_ASSERT(eos_id >= 0, "EOS token ID is valid");
    TEST_ASSERT(pad_id >= 0, "PAD token ID is valid");
    printf("  Special tokens - BOS: %d, EOS: %d, PAD: %d\n", bos_id, eos_id, pad_id);
    
    tokenizer_free(tokenizer);
    
    /* Test loading invalid file */
    tokenizer = tokenizer_create();
    result = tokenizer_load(tokenizer, "nonexistent_file.bin");
    TEST_ASSERT(result == -1, "tokenizer_load() fails for nonexistent file");
    
    tokenizer_free(tokenizer);
}

/**
 * @brief Test basic text encoding
 */
void test_basic_encoding(Tokenizer* tokenizer) {
    printf("\n=== Testing Basic Encoding ===\n");
    
    /* Test simple text */
    const char* test_text = "Hello world";
    TokenizerResult result;
    
    int ret = tokenizer_encode(tokenizer, test_text, &result);
    TEST_ASSERT(ret == 0, "encode simple text succeeds");
    TEST_ASSERT(result.count > 0, "encoded tokens count > 0");
    TEST_ASSERT(result.token_ids != NULL, "token_ids array allocated");
    
    printf("  Text: \"%s\" -> %d tokens: [", test_text, result.count);
    for (int i = 0; i < result.count; i++) {
        printf("%d", result.token_ids[i]);
        if (i < result.count - 1) printf(", ");
    }
    printf("]\n");
    
    tokenizer_result_free(&result);
    
    /* Test empty string */
    ret = tokenizer_encode(tokenizer, "", &result);
    TEST_ASSERT(ret == 0, "encode empty string succeeds");
    TEST_ASSERT(result.count == 0, "empty string produces 0 tokens");
    TEST_ASSERT(result.token_ids == NULL, "empty result has NULL token_ids");
    
    tokenizer_result_free(&result);
    
    /* Test Unicode text */
    const char* unicode_text = "你好世界";  /* "Hello world" in Chinese */
    ret = tokenizer_encode(tokenizer, unicode_text, &result);
    TEST_ASSERT(ret == 0, "encode Unicode text succeeds");
    TEST_ASSERT(result.count > 0, "Unicode text produces tokens");
    
    printf("  Unicode: \"%s\" -> %d tokens\n", unicode_text, result.count);
    tokenizer_result_free(&result);
    
    /* Test longer text */
    const char* long_text = "This is a longer piece of text that should be tokenized into multiple tokens. "
                           "It contains various punctuation marks, numbers like 123 and 456, and different words.";
    ret = tokenizer_encode(tokenizer, long_text, &result);
    TEST_ASSERT(ret == 0, "encode long text succeeds");
    TEST_ASSERT(result.count > 10, "long text produces many tokens");
    
    printf("  Long text (%zu chars) -> %d tokens\n", strlen(long_text), result.count);
    tokenizer_result_free(&result);
}

/**
 * @brief Test token decoding
 */
void test_basic_decoding(Tokenizer* tokenizer) {
    printf("\n=== Testing Basic Decoding ===\n");
    
    /* First encode some text, then decode it back */
    const char* original_text = "The quick brown fox jumps over the lazy dog.";
    TokenizerResult encode_result;
    
    int ret = tokenizer_encode(tokenizer, original_text, &encode_result);
    TEST_ASSERT(ret == 0, "encode for roundtrip test succeeds");
    
    /* Now decode the tokens */
    TokenizerResult decode_result;
    ret = tokenizer_decode(tokenizer, encode_result.token_ids, encode_result.count, &decode_result);
    TEST_ASSERT(ret == 0, "decode tokens succeeds");
    TEST_ASSERT(decode_result.count == encode_result.count, "decode count matches encode count");
    TEST_ASSERT(decode_result.token_strings != NULL, "token_strings array allocated");
    
    /* Print decoded tokens */
    printf("  Decoded tokens:\n");
    for (int i = 0; i < decode_result.count; i++) {
        printf("    [%d]: \"%s\"\n", encode_result.token_ids[i], decode_result.token_strings[i]);
    }
    
    /* Reconstruct text by concatenating tokens */
    int total_len = 1;  /* for null terminator */
    for (int i = 0; i < decode_result.count; i++) {
        total_len += strlen(decode_result.token_strings[i]);
    }
    
    char* reconstructed = malloc(total_len);
    TEST_ASSERT(reconstructed != NULL, "memory allocation for reconstruction succeeds");
    
    reconstructed[0] = '\0';
    for (int i = 0; i < decode_result.count; i++) {
        strcat(reconstructed, decode_result.token_strings[i]);
    }
    
    printf("  Original:      \"%s\"\n", original_text);
    printf("  Reconstructed: \"%s\"\n", reconstructed);
    
    /* Note: Perfect roundtrip might not always work due to BPE encoding,
     * but the lengths should be similar for most text */
    TEST_ASSERT(abs((int)strlen(original_text) - (int)strlen(reconstructed)) <= 5, 
                "reconstructed text length is close to original");
    
    free(reconstructed);
    tokenizer_result_free(&encode_result);
    tokenizer_result_free(&decode_result);
    
    /* Test decoding empty array */
    ret = tokenizer_decode(tokenizer, NULL, 0, &decode_result);
    TEST_ASSERT(ret == 0, "decode empty array succeeds");
    TEST_ASSERT(decode_result.count == 0, "empty decode produces 0 tokens");
    TEST_ASSERT(decode_result.token_strings == NULL, "empty result has NULL token_strings");
    
    tokenizer_result_free(&decode_result);
}

/**
 * @brief Test chat template functionality
 */
void test_chat_template(Tokenizer* tokenizer) {
    printf("\n=== Testing Chat Template ===\n");
    
    const char* user_message = "What is the capital of France?";
    char chat_result[1024];
    
    int ret = tokenizer_wrap_chat(tokenizer, user_message, chat_result, sizeof(chat_result));
    TEST_ASSERT(ret == 0, "wrap_chat succeeds");
    
    printf("  User message: \"%s\"\n", user_message);
    printf("  Chat template result:\n%s\n", chat_result);
    
    /* Check that result contains expected elements */
    TEST_ASSERT(strstr(chat_result, "<|im_start|>user") != NULL, "contains user start token");
    TEST_ASSERT(strstr(chat_result, "<|im_end|>") != NULL, "contains end token");
    TEST_ASSERT(strstr(chat_result, "<|im_start|>assistant") != NULL, "contains assistant start token");
    TEST_ASSERT(strstr(chat_result, user_message) != NULL, "contains original message");
    
    /* Test buffer too small */
    char small_buffer[10];
    ret = tokenizer_wrap_chat(tokenizer, user_message, small_buffer, sizeof(small_buffer));
    TEST_ASSERT(ret == -1, "wrap_chat fails with small buffer");
    
    /* Test empty message */
    ret = tokenizer_wrap_chat(tokenizer, "", chat_result, sizeof(chat_result));
    TEST_ASSERT(ret == 0, "wrap_chat succeeds with empty message");
    TEST_ASSERT(strstr(chat_result, "<|im_start|>user") != NULL, "empty message still has template structure");
}

/**
 * @brief Test special token handling
 */
void test_special_tokens(Tokenizer* tokenizer) {
    printf("\n=== Testing Special Tokens ===\n");
    
    /* Get special token IDs */
    int bos_id, eos_id, pad_id;
    int ret = tokenizer_get_special_tokens(tokenizer, &bos_id, &eos_id, &pad_id);
    TEST_ASSERT(ret == 0, "get_special_tokens succeeds");
    
    /* Test decoding special tokens */
    int special_tokens[] = {bos_id, eos_id, pad_id};
    TokenizerResult result;
    
    ret = tokenizer_decode(tokenizer, special_tokens, 3, &result);
    TEST_ASSERT(ret == 0, "decode special tokens succeeds");
    TEST_ASSERT(result.count == 3, "decode produces 3 tokens");
    
    printf("  Special token strings:\n");
    for (int i = 0; i < result.count; i++) {
        printf("    ID %d -> \"%s\"\n", special_tokens[i], result.token_strings[i]);
    }
    
    /* Special tokens should contain angle brackets (typical format) */
    int has_angle_brackets = 0;
    for (int i = 0; i < result.count; i++) {
        if (strchr(result.token_strings[i], '<') && strchr(result.token_strings[i], '>')) {
            has_angle_brackets++;
        }
    }
    TEST_ASSERT(has_angle_brackets > 0, "at least one special token has angle bracket format");
    
    tokenizer_result_free(&result);
    
    /* Test encoding text with special tokens */
    const char* text_with_specials = "Hello <|im_start|>user world<|im_end|>";
    ret = tokenizer_encode(tokenizer, text_with_specials, &result);
    TEST_ASSERT(ret == 0, "encode text with special tokens succeeds");
    TEST_ASSERT(result.count > 3, "text with specials produces multiple tokens");
    
    printf("  Text with specials: \"%s\" -> %d tokens\n", text_with_specials, result.count);
    tokenizer_result_free(&result);
}

/**
 * @brief Test error conditions and edge cases
 */
void test_error_conditions(Tokenizer* tokenizer) {
    printf("\n=== Testing Error Conditions ===\n");
    
    TokenizerResult result;
    
    /* Test NULL tokenizer */
    int ret = tokenizer_encode(NULL, "test", &result);
    TEST_ASSERT(ret == -1, "encode with NULL tokenizer fails");
    
    ret = tokenizer_decode(NULL, NULL, 0, &result);
    TEST_ASSERT(ret == -1, "decode with NULL tokenizer fails");
    
    /* Test NULL result pointer */
    ret = tokenizer_encode(tokenizer, "test", NULL);
    TEST_ASSERT(ret == -1, "encode with NULL result fails");
    
    /* Test invalid token IDs for decoding */
    int invalid_tokens[] = {-1, 999999999};  /* Negative and very large token IDs */
    ret = tokenizer_decode(tokenizer, invalid_tokens, 2, &result);
    TEST_ASSERT(ret == -1, "decode with invalid token IDs fails");
    
    /* Test NULL text for encoding */
    ret = tokenizer_encode(tokenizer, NULL, &result);
    TEST_ASSERT(ret == -1, "encode with NULL text fails");
    
    /* Test negative token count for decoding */
    ret = tokenizer_decode(tokenizer, NULL, -1, &result);
    TEST_ASSERT(ret == -1, "decode with negative count fails");
    
    /* Test chat wrap with NULL inputs */
    char buffer[100];
    ret = tokenizer_wrap_chat(NULL, "test", buffer, sizeof(buffer));
    TEST_ASSERT(ret == -1, "wrap_chat with NULL tokenizer fails");
    
    ret = tokenizer_wrap_chat(tokenizer, NULL, buffer, sizeof(buffer));
    TEST_ASSERT(ret == -1, "wrap_chat with NULL message fails");
    
    ret = tokenizer_wrap_chat(tokenizer, "test", NULL, sizeof(buffer));
    TEST_ASSERT(ret == -1, "wrap_chat with NULL buffer fails");
    
    /* Test vocab_size with NULL tokenizer */
    int vocab_size = tokenizer_vocab_size(NULL);
    TEST_ASSERT(vocab_size == -1, "vocab_size with NULL tokenizer returns -1");
}

/**
 * @brief Test performance with larger text
 */
void test_performance(Tokenizer* tokenizer) {
    printf("\n=== Testing Performance ===\n");
    
    /* Generate a reasonably large text */
    const char* base_text = "This is a performance test with repeating text. ";
    int base_len = strlen(base_text);
    int repetitions = 100;  /* About 5KB of text */
    
    char* large_text = malloc((base_len * repetitions) + 1);
    TEST_ASSERT(large_text != NULL, "memory allocation for performance test succeeds");
    
    large_text[0] = '\0';
    for (int i = 0; i < repetitions; i++) {
        strcat(large_text, base_text);
    }
    
    printf("  Testing with %zu characters of text\n", strlen(large_text));
    
    /* Test encoding performance */
    TokenizerResult result;
    int ret = tokenizer_encode(tokenizer, large_text, &result);
    TEST_ASSERT(ret == 0, "encode large text succeeds");
    TEST_ASSERT(result.count > repetitions, "large text produces many tokens");
    
    printf("  Encoded %zu chars -> %d tokens\n", strlen(large_text), result.count);
    
    /* Test decoding performance */
    TokenizerResult decode_result;
    ret = tokenizer_decode(tokenizer, result.token_ids, result.count, &decode_result);
    TEST_ASSERT(ret == 0, "decode large token array succeeds");
    TEST_ASSERT(decode_result.count == result.count, "decode count matches");
    
    printf("  Decoded %d tokens -> %d strings\n", result.count, decode_result.count);
    
    free(large_text);
    tokenizer_result_free(&result);
    tokenizer_result_free(&decode_result);
}

/**
 * @brief Main test runner
 */
int main(int argc, char* argv[]) {
    printf("Qwen3 Tokenizer Test Suite\n");
    printf("=========================\n");
    
    if (argc != 2) {
        printf("Usage: %s <tokenizer_binary_file>\n", argv[0]);
        printf("\nFirst run the Python export script:\n");
        printf("  python3 export_qwen3_tokenizer.py\n");
        printf("Then run tests:\n");
        printf("  %s qwen3_tokenizer.bin\n", argv[0]);
        return 1;
    }
    
    const char* tokenizer_file = argv[1];
    
    /* Run lifecycle tests (no loading required) */
    test_tokenizer_lifecycle();
    
    /* Test loading */
    test_tokenizer_loading(tokenizer_file);
    
    /* Load tokenizer for remaining tests */
    Tokenizer* tokenizer = tokenizer_create();
    if (!tokenizer) {
        printf("ERROR: Failed to create tokenizer\n");
        return 1;
    }
    
    if (tokenizer_load(tokenizer, tokenizer_file) != 0) {
        printf("ERROR: Failed to load tokenizer from %s\n", tokenizer_file);
        printf("Make sure to run the export script first:\n");
        printf("  python3 export_qwen3_tokenizer.py\n");
        tokenizer_free(tokenizer);
        return 1;
    }
    
    /* Run main test suite */
    test_basic_encoding(tokenizer);
    test_basic_decoding(tokenizer);
    test_chat_template(tokenizer);
    test_special_tokens(tokenizer);
    test_error_conditions(tokenizer);
    test_performance(tokenizer);
    
    /* Cleanup */
    tokenizer_free(tokenizer);
    
    /* Print summary */
    printf("\n=========================\n");
    printf("Test Results Summary\n");
    printf("=========================\n");
    printf("Tests passed: %d\n", tests_passed);
    printf("Tests failed: %d\n", tests_failed);
    printf("Total tests:  %d\n", tests_passed + tests_failed);
    
    if (tests_failed == 0) {
        printf("\n🎉 ALL TESTS PASSED! 🎉\n");
        return 0;
    } else {
        printf("\n❌ %d TESTS FAILED ❌\n", tests_failed);
        return 1;
    }
}