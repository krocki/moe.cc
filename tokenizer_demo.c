#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Simple demonstration of Qwen3 tokenizer usage
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <tokenizer_binary_file>\n", argv[0]);
        printf("Example: %s qwen3_tokenizer.bin\n", argv[0]);
        return 1;
    }
    
    const char* tokenizer_file = argv[1];
    
    /* Create and load tokenizer */
    printf("Loading Qwen3 tokenizer from %s...\n", tokenizer_file);
    Tokenizer* tokenizer = tokenizer_create();
    if (!tokenizer) {
        printf("ERROR: Failed to create tokenizer\n");
        return 1;
    }
    
    if (tokenizer_load(tokenizer, tokenizer_file) != 0) {
        printf("ERROR: Failed to load tokenizer\n");
        tokenizer_free(tokenizer);
        return 1;
    }
    
    printf("✓ Tokenizer loaded successfully\n");
    printf("✓ Vocabulary size: %d tokens\n", tokenizer_vocab_size(tokenizer));
    
    /* Get special tokens */
    int bos_id, eos_id, pad_id;
    tokenizer_get_special_tokens(tokenizer, &bos_id, &eos_id, &pad_id);
    printf("✓ Special tokens: BOS=%d, EOS=%d, PAD=%d\n\n", bos_id, eos_id, pad_id);
    
    /* Demonstrate encoding */
    const char* test_texts[] = {
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "你好，世界！这是一个测试。",
        "What is the capital of France? I'd like to know more about Paris.",
        ""
    };
    
    printf("=== ENCODING DEMONSTRATION ===\n");
    for (int i = 0; test_texts[i][0] != '\0'; i++) {
        const char* text = test_texts[i];
        TokenizerResult result;
        
        if (tokenizer_encode(tokenizer, text, &result) == 0) {
            printf("Text: \"%s\"\n", text);
            printf("Tokens (%d): [", result.count);
            for (int j = 0; j < result.count; j++) {
                printf("%d", result.token_ids[j]);
                if (j < result.count - 1) printf(", ");
            }
            printf("]\n\n");
            tokenizer_result_free(&result);
        } else {
            printf("ERROR: Failed to encode text: \"%s\"\n", text);
        }
    }
    
    /* Demonstrate decoding */
    printf("=== DECODING DEMONSTRATION ===\n");
    const char* demo_text = "Artificial intelligence is transforming the world.";
    TokenizerResult encode_result;
    
    if (tokenizer_encode(tokenizer, demo_text, &encode_result) == 0) {
        printf("Original: \"%s\"\n", demo_text);
        
        TokenizerResult decode_result;
        if (tokenizer_decode(tokenizer, encode_result.token_ids, encode_result.count, &decode_result) == 0) {
            printf("Token breakdown:\n");
            for (int i = 0; i < decode_result.count; i++) {
                printf("  [%d] -> \"%s\"\n", encode_result.token_ids[i], decode_result.token_strings[i]);
            }
            
            /* Reconstruct text */
            printf("\nReconstructed: \"");
            for (int i = 0; i < decode_result.count; i++) {
                printf("%s", decode_result.token_strings[i]);
            }
            printf("\"\n\n");
            
            tokenizer_result_free(&decode_result);
        } else {
            printf("ERROR: Failed to decode tokens\n");
        }
        
        tokenizer_result_free(&encode_result);
    }
    
    /* Demonstrate chat template */
    printf("=== CHAT TEMPLATE DEMONSTRATION ===\n");
    const char* user_messages[] = {
        "Hello! How are you?",
        "What is machine learning?",
        "Can you help me write a Python function?",
        ""
    };
    
    for (int i = 0; user_messages[i][0] != '\0'; i++) {
        const char* message = user_messages[i];
        char chat_formatted[2048];
        
        if (tokenizer_wrap_chat(tokenizer, message, chat_formatted, sizeof(chat_formatted)) == 0) {
            printf("User message: \"%s\"\n", message);
            printf("Chat format:\n%s\n", chat_formatted);
            printf("---\n");
        } else {
            printf("ERROR: Failed to format chat for message: \"%s\"\n", message);
        }
    }
    
    /* Demonstrate special token decoding */
    printf("=== SPECIAL TOKEN DEMONSTRATION ===\n");
    int special_tokens[] = {bos_id, eos_id};
    TokenizerResult special_result;
    
    if (tokenizer_decode(tokenizer, special_tokens, 2, &special_result) == 0) {
        printf("Special token strings:\n");
        for (int i = 0; i < special_result.count; i++) {
            printf("  ID %d -> \"%s\"\n", special_tokens[i], special_result.token_strings[i]);
        }
        tokenizer_result_free(&special_result);
    }
    
    /* Clean up */
    tokenizer_free(tokenizer);
    printf("\n✓ Demo completed successfully!\n");
    
    return 0;
}