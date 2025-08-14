# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -g -O2 -Wall #-DDEBUG -DBENCH

OBJS = io.o utils.o kernels.o

all: test_embed test_head test_layer_trace test_model_trace test_expert test_moe_block test_rmsnorm test_attn test_rope list_bin

test_expert: test_expert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_moe_block: test_moe_block.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_rmsnorm: test_rmsnorm.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_attn: test_attn.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_rope: test_rope.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_layer: test_layer.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_embed: test_embed.o io.o utils.o
	$(CC) -O2 -Wall -o $@ $^ -lm

test_stack: test_stack.o io.o utils.o kernels.o
	$(CC) -O2 -Wall -o $@ $^ -lm

test_head: test_head.o io.o utils.o kernels.o
	$(CC) -O2 -Wall -o $@ $^ -lm

test-model-v0: test-model-v0.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_model: test_model.o io.o utils.o kernels.o model.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_layer_trace: test_layer_trace.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_model_trace: test_model_trace.o io.o utils.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

list_bin: list_bin.o io.o
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_expert test_moe_block
