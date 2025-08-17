# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -g -O2 -Wall -DDEBUG -DBENCH

OBJS = io.o utils.o

all: list_bin test_model_trace tokenizer_demo tokenizer_test

test_model_trace: test_model_trace.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Export tokenizer from Qwen3 model
export_tokenizer:
	python3 export_qwen3_tokenizer.py

list_bin: list_bin.o io.o
	$(CC) $(CFLAGS) -o $@ $^

# Tokenizer targets
tokenizer_test: tokenizer_test.o tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^

tokenizer_demo: tokenizer_demo.o tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_expert test_moe_block tokenizer_test tokenizer_demo test_quantization_consistency test_model_loading test_group_size_edge_cases list_bin test_model_trace
	$(RM) -f qwen3_tokenizer.bin qwen3_tokenizer_meta.json
