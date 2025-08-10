# Makefile (tab-indented rules)
CC = gcc
CFLAGS = -O2 -Wall

all: test_expert

test_expert: test_expert.o io.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c io.h
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_expert

