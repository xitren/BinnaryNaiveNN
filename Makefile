testBIN = test

CFLAGS = -std=c99 -Wall -Wextra -pedantic -Ofast -flto -march=native

INCLUDES = -Iref -Isrc -I.

LDFLAGS = -lm -Os

CC = gcc

SRCREF = ref/Tinn.c src/logger.c src/data_reader.c
SRC = src/neuron.c src/logger.c src/data_reader.c
SRCBNN = src/bnn.c src/logger.c src/data_reader.c src/genetic_search.c

all:
	$(CC) -o $(testBIN) $(SRC) $(CFLAGS) $(LDFLAGS)

test_ref_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_ref.c $(SRCREF) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_src_learn:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_learn_src.c $(SRC) -DINPUTS=1400 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/2_work_learn_src.c $(SRC) -DINPUTS=700 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/3_work_learn_src.c $(SRC) -DINPUTS=200 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/4_work_learn_src.c $(SRC) -DINPUTS=140 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/5_work_learn_src.c $(SRC) -DINPUTS=20 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_src_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/2_work_src.c $(SRC) -DINPUTS=1400 $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
test_bnn_work:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/1_work_bnn.c $(SRCBNN) $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)
	
prepare_bitcnt_table:
	rm -f $(testBIN)
	$(CC) -o $(testBIN) tests/bitcnt_table.c src/logger.c $(CFLAGS) $(INCLUDES) $(LDFLAGS)
	./$(testBIN)
	rm -f $(testBIN)

clean:
	rm -f $(testBIN)
