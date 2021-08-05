testBIN = test

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

INCLUDES = -Iref -Isrc -I.

LDFLAGS = -lm

CC = gcc

SRCREF = ref/Tinn.c
SRC = src/neuron.c

all:
	$(CC) -o $(testBIN) $(SRC) $(CFLAGS) $(LDFLAGS)

test_ref_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_ref.c $(SRCREF) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_src_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_src.c $(SRCREF) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)

clean:
	rm -f $(testBIN)
