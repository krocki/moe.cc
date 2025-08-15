#!/usr/bin/env python3
"""
Qwen3 Tokenizer Export Script
Exports Qwen3-30B-A3B tokenizer data to a binary format for C++ consumption.
"""

import struct
import json
from transformers import AutoTokenizer


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings.
    Specifically avoids mapping to the BMP whitespace/control characters
    the BPE code barfs on (like '\0').
    """
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def export_qwen3_tokenizer(model_name="Qwen/Qwen3-30B-A3B", output_file="qwen3_tokenizer"):
    """
    Export Qwen3 tokenizer to binary format for C++ usage.
    """
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the tokenizer data structure
    tokenizer_data = json.loads(tokenizer.backend_tokenizer.to_str())
    
    # Extract vocab and merges
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]
    
    # Build token ID to token string mapping
    id_to_token = {v: k for k, v in vocab.items()}
    max_token_id = max(id_to_token.keys())
    
    # Get byte encoder mapping
    byte_encoder = bytes_to_unicode()
    
    # Get special tokens and their strings
    special_tokens = {
        "bos_token": tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 151643,
        "eos_token": tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 151645,
        "pad_token": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643,
    }
    
    # Add special token strings to id_to_token mapping
    if tokenizer.bos_token:
        max_token_id = max(max_token_id, special_tokens["bos_token"])
        id_to_token[special_tokens["bos_token"]] = tokenizer.bos_token
    
    if tokenizer.eos_token:
        max_token_id = max(max_token_id, special_tokens["eos_token"])
        id_to_token[special_tokens["eos_token"]] = tokenizer.eos_token
    
    if tokenizer.pad_token:
        max_token_id = max(max_token_id, special_tokens["pad_token"])
        id_to_token[special_tokens["pad_token"]] = tokenizer.pad_token
    
    # Chat template
    chat_template = """<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
"""
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"BOS token ID: {special_tokens['bos_token']}")
    print(f"EOS token ID: {special_tokens['eos_token']}")
    
    # Write binary format
    with open(f"{output_file}.bin", "wb") as f:
        # Header: Magic number and version
        f.write(b"QW3T")  # Magic: Qwen3 Tokenizer
        f.write(struct.pack("<I", 1))  # Version
        
        # Vocab size and max token ID
        f.write(struct.pack("<I", len(vocab)))
        f.write(struct.pack("<I", max_token_id))
        
        # Special tokens
        f.write(struct.pack("<I", special_tokens["bos_token"]))
        f.write(struct.pack("<I", special_tokens["eos_token"]))
        f.write(struct.pack("<I", special_tokens["pad_token"]))
        
        # Chat template
        chat_bytes = chat_template.encode("utf-8")
        f.write(struct.pack("<I", len(chat_bytes)))
        f.write(chat_bytes)
        
        # Byte encoder mapping (256 entries)
        for i in range(256):
            encoded_char = byte_encoder[i].encode("utf-8")
            f.write(struct.pack("<I", len(encoded_char)))
            f.write(encoded_char)
        
        # Vocabulary: for each token ID, write the token string
        for token_id in range(max_token_id + 1):
            if token_id in id_to_token:
                token_str = id_to_token[token_id]
                token_bytes = token_str.encode("utf-8")
                f.write(struct.pack("<I", len(token_bytes)))
                f.write(token_bytes)
            else:
                # Empty token
                f.write(struct.pack("<I", 0))
        
        # Merge rules: write number of merges, then each merge rule
        f.write(struct.pack("<I", len(merges)))
        for merge in merges:
            if isinstance(merge, list):
                merge_str = " ".join(merge)
            else:
                merge_str = merge
            merge_bytes = merge_str.encode("utf-8")
            f.write(struct.pack("<I", len(merge_bytes)))
            f.write(merge_bytes)
    
    # Write metadata as JSON for debugging/inspection
    metadata = {
        "vocab_size": len(vocab),
        "max_token_id": max_token_id,
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "chat_template": chat_template,
        "sample_tokens": {str(k): v for k, v in list(id_to_token.items())[:10]}
    }
    
    with open(f"{output_file}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Exported tokenizer to {output_file}.bin")
    print(f"Metadata saved to {output_file}_meta.json")


if __name__ == "__main__":
    export_qwen3_tokenizer()
