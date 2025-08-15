#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/**
 * @brief Hash table entry for string to ID mapping
 */
typedef struct HashEntry {
    char* key;              /** Token string (encoded) */
    int value;              /** Token ID */
    struct HashEntry* next; /** Next entry in collision chain */
} HashEntry;

/**
 * @brief Hash table for fast token lookup
 */
typedef struct {
    HashEntry** buckets;    /** Array of bucket pointers */
    int bucket_count;       /** Number of buckets */
    int size;               /** Number of entries */
} HashTable;

/**
 * @brief Trie node for prefix matching during encoding
 */
typedef struct TrieNode {
    struct TrieNode* children[256];  /** Child nodes for each byte value */
    int token_id;                    /** Token ID if this is a complete token (-1 if not) */
} TrieNode;

/**
 * @brief BPE merge rule
 */
typedef struct {
    char* pattern;          /** Merge pattern (e.g., "a b") */
    int rank;               /** Merge priority (lower = higher priority) */
} MergeRule;

/**
 * @brief Main tokenizer structure (opaque to users)
 */
struct Tokenizer {
    /* Vocabulary and mappings */
    HashTable* token_to_id;         /** String to ID mapping for encoding */
    char** id_to_token;             /** ID to string mapping for decoding */
    int vocab_size;                 /** Total vocabulary size */
    int max_token_id;               /** Highest token ID */
    
    /* Byte encoding tables (GPT-2 style) */
    char* byte_encoder[256];        /** Byte to encoded string mapping */
    HashTable* byte_decoder;        /** Encoded string to byte mapping */
    
    /* BPE merge rules */
    MergeRule* merge_rules;         /** Array of merge rules */
    int num_merges;                 /** Number of merge rules */
    
    /* Special tokens */
    int bos_token_id;               /** Beginning of sequence token */
    int eos_token_id;               /** End of sequence token */
    int pad_token_id;               /** Padding token */
    
    /* Chat template */
    char chat_template[MAX_CHAT_TEMPLATE_LENGTH]; /** Chat formatting template */
    
    /* Trie for efficient token matching */
    TrieNode* trie_root;            /** Root of prefix matching trie */
};

/* ===== HASH TABLE IMPLEMENTATION ===== */

/**
 * @brief Simple hash function for strings
 */
static unsigned int hash_string(const char* str, int bucket_count) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash % bucket_count;
}

/**
 * @brief Create new hash table
 */
static HashTable* hash_table_create(int bucket_count) {
    HashTable* table = malloc(sizeof(HashTable));
    if (!table) return NULL;
    
    table->buckets = calloc(bucket_count, sizeof(HashEntry*));
    if (!table->buckets) {
        free(table);
        return NULL;
    }
    
    table->bucket_count = bucket_count;
    table->size = 0;
    return table;
}

/**
 * @brief Insert key-value pair into hash table
 */
static int hash_table_put(HashTable* table, const char* key, int value) {
    unsigned int bucket = hash_string(key, table->bucket_count);
    
    /* Check if key already exists */
    HashEntry* entry = table->buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;  /* Update existing */
            return 0;
        }
        entry = entry->next;
    }
    
    /* Create new entry */
    entry = malloc(sizeof(HashEntry));
    if (!entry) return -1;
    
    entry->key = malloc(strlen(key) + 1);
    if (!entry->key) {
        free(entry);
        return -1;
    }
    strcpy(entry->key, key);
    entry->value = value;
    entry->next = table->buckets[bucket];
    table->buckets[bucket] = entry;
    table->size++;
    
    return 0;
}

/**
 * @brief Get value by key from hash table
 */
static int hash_table_get(HashTable* table, const char* key, int* value) {
    unsigned int bucket = hash_string(key, table->bucket_count);
    
    HashEntry* entry = table->buckets[bucket];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            *value = entry->value;
            return 0;  /* Found */
        }
        entry = entry->next;
    }
    
    return -1;  /* Not found */
}

/**
 * @brief Free hash table memory
 */
static void hash_table_free(HashTable* table) {
    if (!table) return;
    
    for (int i = 0; i < table->bucket_count; i++) {
        HashEntry* entry = table->buckets[i];
        while (entry) {
            HashEntry* next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    
    free(table->buckets);
    free(table);
}

/* ===== TRIE IMPLEMENTATION ===== */

/**
 * @brief Create new trie node
 */
static TrieNode* trie_node_create(void) {
    TrieNode* node = malloc(sizeof(TrieNode));
    if (!node) return NULL;
    
    memset(node->children, 0, sizeof(node->children));
    node->token_id = -1;
    return node;
}

/**
 * @brief Insert token into trie
 */
static int trie_insert(TrieNode* root, const char* token, int token_id) {
    TrieNode* current = root;
    
    for (const char* p = token; *p; p++) {
        unsigned char byte = (unsigned char)*p;
        
        if (!current->children[byte]) {
            current->children[byte] = trie_node_create();
            if (!current->children[byte]) return -1;
        }
        
        current = current->children[byte];
    }
    
    current->token_id = token_id;
    return 0;
}

/**
 * @brief Free trie memory recursively
 */
static void trie_free(TrieNode* node) {
    if (!node) return;
    
    for (int i = 0; i < 256; i++) {
        trie_free(node->children[i]);
    }
    
    free(node);
}

/* ===== BYTE ENCODING (GPT-2 style) ===== */

/**
 * @brief Initialize GPT-2 style byte encoding
 */
static int init_byte_encoding(Tokenizer* tokenizer, char byte_encoder_data[256][8]) {
    tokenizer->byte_decoder = hash_table_create(512);
    if (!tokenizer->byte_decoder) return -1;
    
    for (int i = 0; i < 256; i++) {
        tokenizer->byte_encoder[i] = malloc(strlen(byte_encoder_data[i]) + 1);
        if (!tokenizer->byte_encoder[i]) return -1;
        
        strcpy(tokenizer->byte_encoder[i], byte_encoder_data[i]);
        
        /* Build reverse mapping */
        hash_table_put(tokenizer->byte_decoder, byte_encoder_data[i], i);
    }
    
    return 0;
}

/**
 * @brief Convert raw bytes to encoded string
 */
static char* bytes_to_encoded(Tokenizer* tokenizer, const char* input, int input_len) {
    int max_output_len = input_len * 4 + 1;  /* Conservative estimate */
    char* output = malloc(max_output_len);
    if (!output) return NULL;
    
    int pos = 0;
    for (int i = 0; i < input_len; i++) {
        unsigned char byte = (unsigned char)input[i];
        const char* encoded = tokenizer->byte_encoder[byte];
        int encoded_len = strlen(encoded);
        
        if (pos + encoded_len >= max_output_len - 1) {
            /* Realloc if needed */
            max_output_len *= 2;
            char* new_output = realloc(output, max_output_len);
            if (!new_output) {
                free(output);
                return NULL;
            }
            output = new_output;
        }
        
        strcpy(output + pos, encoded);
        pos += encoded_len;
    }
    
    output[pos] = '\0';
    return output;
}

/**
 * @brief Convert encoded string back to raw bytes
 */
static char* encoded_to_bytes(Tokenizer* tokenizer, const char* encoded) {
    int max_output_len = strlen(encoded) + 1;
    char* output = malloc(max_output_len);
    if (!output) return NULL;
    
    int output_pos = 0;
    const char* p = encoded;
    
    while (*p) {
        /* Find the longest matching encoded sequence */
        int matched_byte = -1;
        int max_match_len = 0;
        
        for (int len = 1; len <= 4 && p[len-1]; len++) {
            char temp[5];
            strncpy(temp, p, len);
            temp[len] = '\0';
            
            int byte_val;
            if (hash_table_get(tokenizer->byte_decoder, temp, &byte_val) == 0) {
                if (len > max_match_len) {
                    max_match_len = len;
                    matched_byte = byte_val;
                }
            }
        }
        
        if (matched_byte >= 0) {
            output[output_pos++] = (char)matched_byte;
            p += max_match_len;
        } else {
            /* Fallback: copy as-is */
            output[output_pos++] = *p++;
        }
    }
    
    output[output_pos] = '\0';
    return output;
}

/* ===== MAIN TOKENIZER FUNCTIONS ===== */

Tokenizer* tokenizer_create(void) {
    Tokenizer* tokenizer = malloc(sizeof(Tokenizer));
    if (!tokenizer) return NULL;
    
    memset(tokenizer, 0, sizeof(Tokenizer));
    
    tokenizer->token_to_id = hash_table_create(50000);
    if (!tokenizer->token_to_id) {
        free(tokenizer);
        return NULL;
    }
    
    tokenizer->trie_root = trie_node_create();
    if (!tokenizer->trie_root) {
        hash_table_free(tokenizer->token_to_id);
        free(tokenizer);
        return NULL;
    }
    
    return tokenizer;
}

void tokenizer_free(Tokenizer* tokenizer) {
    if (!tokenizer) return;
    
    hash_table_free(tokenizer->token_to_id);
    hash_table_free(tokenizer->byte_decoder);
    
    if (tokenizer->id_to_token) {
        for (int i = 0; i <= tokenizer->max_token_id; i++) {
            free(tokenizer->id_to_token[i]);
        }
        free(tokenizer->id_to_token);
    }
    
    for (int i = 0; i < 256; i++) {
        free(tokenizer->byte_encoder[i]);
    }
    
    for (int i = 0; i < tokenizer->num_merges; i++) {
        free(tokenizer->merge_rules[i].pattern);
    }
    free(tokenizer->merge_rules);
    
    trie_free(tokenizer->trie_root);
    free(tokenizer);
}

/**
 * @brief Read 32-bit little-endian integer from file
 */
static int read_uint32(FILE* f, uint32_t* value) {
    if (fread(value, sizeof(uint32_t), 1, f) != 1) return -1;
    return 0;
}

/**
 * @brief Read string from file (length-prefixed)
 */
static char* read_string(FILE* f) {
    uint32_t len;
    if (read_uint32(f, &len) != 0) return NULL;
    if (len > 10000000) return NULL;  /* Sanity check */
    
    char* str = malloc(len + 1);
    if (!str) return NULL;
    
    if (len > 0 && fread(str, 1, len, f) != len) {
        free(str);
        return NULL;
    }
    
    str[len] = '\0';
    return str;
}

int tokenizer_load(Tokenizer* tokenizer, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;
    
    /* Read and verify header */
    uint32_t magic, version;
    if (read_uint32(f, &magic) != 0 || magic != TOKENIZER_MAGIC ||
        read_uint32(f, &version) != 0 || version != TOKENIZER_VERSION) {
        fclose(f);
        return -1;
    }
    
    /* Read vocab info */
    uint32_t vocab_size, max_token_id;
    if (read_uint32(f, &vocab_size) != 0 || read_uint32(f, &max_token_id) != 0) {
        fclose(f);
        return -1;
    }
    
    tokenizer->vocab_size = vocab_size;
    tokenizer->max_token_id = max_token_id;
    
    /* Read special tokens */
    uint32_t bos, eos, pad;
    if (read_uint32(f, &bos) != 0 || read_uint32(f, &eos) != 0 || read_uint32(f, &pad) != 0) {
        fclose(f);
        return -1;
    }
    
    tokenizer->bos_token_id = bos;
    tokenizer->eos_token_id = eos;
    tokenizer->pad_token_id = pad;
    
    /* Read chat template */
    char* chat_template = read_string(f);
    if (!chat_template) {
        fclose(f);
        return -1;
    }
    strncpy(tokenizer->chat_template, chat_template, MAX_CHAT_TEMPLATE_LENGTH - 1);
    tokenizer->chat_template[MAX_CHAT_TEMPLATE_LENGTH - 1] = '\0';
    free(chat_template);
    
    /* Read byte encoder */
    char byte_encoder_data[256][8];
    for (int i = 0; i < 256; i++) {
        char* encoded_char = read_string(f);
        if (!encoded_char) {
            fclose(f);
            return -1;
        }
        strncpy(byte_encoder_data[i], encoded_char, 7);
        byte_encoder_data[i][7] = '\0';
        free(encoded_char);
    }
    
    if (init_byte_encoding(tokenizer, byte_encoder_data) != 0) {
        fclose(f);
        return -1;
    }
    
    /* Read vocabulary */
    tokenizer->id_to_token = calloc(max_token_id + 1, sizeof(char*));
    if (!tokenizer->id_to_token) {
        fclose(f);
        return -1;
    }
    
    for (uint32_t i = 0; i <= max_token_id; i++) {
        char* token = read_string(f);
        if (!token) {
            fclose(f);
            return -1;
        }
        
        if (strlen(token) > 0) {
            tokenizer->id_to_token[i] = token;
            hash_table_put(tokenizer->token_to_id, token, i);
            trie_insert(tokenizer->trie_root, token, i);
        } else {
            free(token);
            tokenizer->id_to_token[i] = NULL;
        }
    }
    
    /* Read merge rules */
    if (read_uint32(f, (uint32_t*)&tokenizer->num_merges) != 0) {
        fclose(f);
        return -1;
    }
    
    tokenizer->merge_rules = malloc(tokenizer->num_merges * sizeof(MergeRule));
    if (!tokenizer->merge_rules && tokenizer->num_merges > 0) {
        fclose(f);
        return -1;
    }
    
    for (int i = 0; i < tokenizer->num_merges; i++) {
        tokenizer->merge_rules[i].pattern = read_string(f);
        tokenizer->merge_rules[i].rank = i;
        if (!tokenizer->merge_rules[i].pattern) {
            fclose(f);
            return -1;
        }
    }
    
    fclose(f);
    return 0;
}

/**
 * @brief Find longest matching token using trie
 */
static int find_longest_token(Tokenizer* tokenizer, const char* text, int start, int len, int* match_len) {
    TrieNode* current = tokenizer->trie_root;
    int last_token_id = -1;
    int last_match_len = 0;
    
    for (int i = start; i < start + len; i++) {
        unsigned char byte = (unsigned char)text[i];
        
        if (!current->children[byte]) break;
        
        current = current->children[byte];
        
        if (current->token_id >= 0) {
            last_token_id = current->token_id;
            last_match_len = i - start + 1;
        }
    }
    
    *match_len = last_match_len;
    return last_token_id;
}

int tokenizer_encode(Tokenizer* tokenizer, const char* text, TokenizerResult* result) {
    if (!tokenizer || !text || !result) return -1;
    
    int text_len = strlen(text);
    if (text_len == 0) {
        result->token_ids = NULL;
        result->count = 0;
        result->capacity = 0;
        return 0;
    }
    
    /* Convert to encoded bytes */
    char* encoded = bytes_to_encoded(tokenizer, text, text_len);
    if (!encoded) return -1;
    
    int encoded_len = strlen(encoded);
    
    /* Allocate result array (conservative estimate) */
    int capacity = encoded_len + 16;
    int* token_ids = malloc(capacity * sizeof(int));
    if (!token_ids) {
        free(encoded);
        return -1;
    }
    
    int token_count = 0;
    int pos = 0;
    
    /* Greedy longest-match tokenization */
    while (pos < encoded_len) {
        int match_len;
        int token_id = find_longest_token(tokenizer, encoded, pos, encoded_len - pos, &match_len);
        
        if (token_id >= 0 && match_len > 0) {
            /* Expand array if needed */
            if (token_count >= capacity) {
                capacity *= 2;
                int* new_array = realloc(token_ids, capacity * sizeof(int));
                if (!new_array) {
                    free(token_ids);
                    free(encoded);
                    return -1;
                }
                token_ids = new_array;
            }
            
            token_ids[token_count++] = token_id;
            pos += match_len;
        } else {
            /* No token found - this shouldn't happen with proper vocabulary */
            free(token_ids);
            free(encoded);
            return -1;
        }
    }
    
    free(encoded);
    
    result->token_ids = token_ids;
    result->token_strings = NULL;
    result->count = token_count;
    result->capacity = capacity;
    
    return 0;
}

int tokenizer_decode(Tokenizer* tokenizer, const int* token_ids, int num_tokens, TokenizerResult* result) {
    if (!tokenizer || !result || num_tokens < 0) return -1;
    
    if (num_tokens == 0 || !token_ids) {
        result->token_strings = NULL;
        result->count = 0;
        result->capacity = 0;
        result->token_ids = NULL;
        return 0;
    }
    
    char** token_strings = malloc(num_tokens * sizeof(char*));
    if (!token_strings) return -1;
    
    for (int i = 0; i < num_tokens; i++) {
        int token_id = token_ids[i];
        
        if (token_id < 0 || token_id > tokenizer->max_token_id || !tokenizer->id_to_token[token_id]) {
            /* Free already allocated strings */
            for (int j = 0; j < i; j++) {
                free(token_strings[j]);
            }
            free(token_strings);
            return -1;
        }
        
        const char* token = tokenizer->id_to_token[token_id];
        
        /* For now, treat all tokens as encoded (BPE tokens) - decode from bytes */
        char* decoded = encoded_to_bytes(tokenizer, token);
        if (!decoded) {
            /* If decoding fails, use token as-is (likely a special token) */
            token_strings[i] = malloc(strlen(token) + 1);
            if (!token_strings[i]) {
                for (int j = 0; j < i; j++) free(token_strings[j]);
                free(token_strings);
                return -1;
            }
            strcpy(token_strings[i], token);
        } else {
            token_strings[i] = decoded;
        }
    }
    
    result->token_ids = NULL;
    result->token_strings = token_strings;
    result->count = num_tokens;
    result->capacity = num_tokens;
    
    return 0;
}

int tokenizer_wrap_chat(Tokenizer* tokenizer, const char* message, char* result, int max_length) {
    if (!tokenizer || !message || !result) return -1;
    
    /* Replace {message} placeholder in chat template */
    const char* placeholder = "{message}";
    char* template_copy = malloc(strlen(tokenizer->chat_template) + 1);
    if (!template_copy) return -1;
    strcpy(template_copy, tokenizer->chat_template);
    
    char* pos = strstr(template_copy, placeholder);
    if (pos) {
        /* Calculate needed length */
        int before_len = pos - template_copy;
        int after_len = strlen(pos + strlen(placeholder));
        int needed_len = before_len + strlen(message) + after_len + 1;
        
        if (needed_len > max_length) {
            free(template_copy);
            return -1;
        }
        
        /* Build result with substitution */
        strncpy(result, template_copy, before_len);
        result[before_len] = '\0';
        strcat(result, message);
        strcat(result, pos + strlen(placeholder));
    } else {
        /* No placeholder, just copy template */
        if ((int)strlen(tokenizer->chat_template) >= max_length) {
            free(template_copy);
            return -1;
        }
        strcpy(result, tokenizer->chat_template);
    }
    
    free(template_copy);
    return 0;
}

void tokenizer_result_free(TokenizerResult* result) {
    if (!result) return;
    
    if (result->token_ids) {
        free(result->token_ids);
        result->token_ids = NULL;
    }
    
    if (result->token_strings) {
        for (int i = 0; i < result->count; i++) {
            free(result->token_strings[i]);
        }
        free(result->token_strings);
        result->token_strings = NULL;
    }
    
    result->count = 0;
    result->capacity = 0;
}

int tokenizer_vocab_size(Tokenizer* tokenizer) {
    return tokenizer ? tokenizer->vocab_size : -1;
}

int tokenizer_get_special_tokens(Tokenizer* tokenizer, int* bos_id, int* eos_id, int* pad_id) {
    if (!tokenizer) return -1;
    
    if (bos_id) *bos_id = tokenizer->bos_token_id;
    if (eos_id) *eos_id = tokenizer->eos_token_id;
    if (pad_id) *pad_id = tokenizer->pad_token_id;
    
    return 0;
}