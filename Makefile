testBIN = test

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

INCLUDES = -Iref -Isrc -I.

LDFLAGS = -lm

CC = gcc

SRCREF = ref/Tinn.c
SRC = src/neuron.c
SRCBNN = src/bnn.c

all:
	$(CC) -o $(testBIN) $(SRC) $(CFLAGS) $(LDFLAGS)

test_ref_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_ref.c $(SRCREF) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_src_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_src.c $(SRC) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_bnn_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_bnn.c $(SRCBNN) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)

clean:
	rm -f $(testBIN)
