# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -g -O2 -Wall -DDEBUG -DBENCH

OBJS = io.o utils.o quant.o

all: list_bin test_model_trace tokenizer_demo tokenizer_test convert

# Main targets
convert: convert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Test targets  
test_convert: test_convert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_convert_integration: test_convert_integration.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_group_quantization: test_group_quantization.o $(OBJS) kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Test runner targets
test: test_convert test_convert_integration test_group_quantization convert
	@echo "Running unit tests..."
	./test_convert
	@echo "Running integration tests..."
	./test_convert_integration
	@echo "Running group quantization tests..."
	./test_group_quantization

test-unit: test_convert
	@echo "Running unit tests..."
	./test_convert

test-integration: test_convert_integration convert
	@echo "Running integration tests..."
	./test_convert_integration

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
	$(RM) -f *.o test_expert test_moe_block tokenizer_test tokenizer_demo test_quantization_consistency test_model_loading test_group_size_edge_cases list_bin test_model_trace convert test_convert test_convert_integration test_group_quantization
	$(RM) -f qwen3_tokenizer.bin qwen3_tokenizer_meta.json
