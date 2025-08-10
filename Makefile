# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -O2 -Wall -DDEBUG -DBENCH

OBJS = io.o utils.o kernels.o

all: test_expert test_moe_block

test_expert: test_expert.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

test_moe_block: test_moe_block.o $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_expert test_moe_block
