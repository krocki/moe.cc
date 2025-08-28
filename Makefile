# Makefile (tab-indented rules)
CC = clang
#CFLAGS = -O3 -fopenmp -march=native -ffast-math -Wall
CFLAGS = -O3 -fopenmp -march=native -ffast-math -Wall

OBJS = io.o utils.o quants/quant.o debug_utils.o profiler.o quants/tensor.o

all: list_bin convert run

# Clean build without profiling overhead - optimized performance
run: run.o io.o io_mmap.o utils.o debug_utils.o tokenizer.o quants/quant.o profiler.o quants/tensor.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Main targets
convert: convert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_model_trace: test_model_trace.o io.o io_mmap.o utils.o debug_utils.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# Export tokenizer from Qwen3 model
export_tokenizer:
	python3 export_qwen3_tokenizer.py

list_bin: list_bin.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

quants/%.o: quants/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -f *.o list_bin test_model_trace convert run
