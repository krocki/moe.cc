# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -O2 -Wall -DDEBUG -DBENCH

OBJS = io.o utils.o kernels.o

all: test_expert test_moe_block test_rmsnorm test_attn list_bin

test_expert: test_expert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_moe_block: test_moe_block.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_rmsnorm: test_rmsnorm.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_attn: test_attn.o io.o utils.o kernels.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

list_bin: list_bin.o io.o
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_expert test_moe_block
