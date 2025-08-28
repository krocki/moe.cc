# Makefile (tab-indented rules)
CC = clang
#CFLAGS = -O3 -fopenmp -march=native -ffast-math -Wall
CFLAGS = -O3 -fopenmp -march=native -ffast-math -Wall

OBJS = quant.o tensor.o

all: test_quant

test_quant: $(OBJS) test_quant.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) -f *.o test_quant
