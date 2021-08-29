#include "bnn.h"
#include "genetic_search.h"
#include "data_reader.h"
#include "logger.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

static data data_reader;
static network bnn;
static group_type** inputs;
static group_type** outputs;

float error_parser(const chromosome_binary* const chr)
{
    size_t err;
    const size_t size = CHROMOSOME_SIZE / BATCH;
    err = nn_error(&bnn, inputs, outputs, data_reader.rows);
    return err;
}

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 255;
    const int nops = 10;
    population_ranger pranger;
    
    // Load the training set.
    DESCRIBE_LOG("Read started\n");
    build_data(&data_reader, "tests/semeion.data", " ", nips, nops);
    DESCRIBE_LOG("Files readed\n");
    
    DESCRIBE_LOG("Initialization started\n");
    nn_initialize(&bnn);
    initiate_population_ranger(&pranger, &error_parser, POP_MAX, 0.01, 0.6);
    DESCRIBE_LOG("Initialization ended\n");
    
    DESCRIBE_LOG("Data conversion\n");
    inputs = (group_type**) malloc((data_reader.rows) * sizeof(group_type*));
    if (!inputs)
        ERROR_LOG("Inputs memory allocation error!\n");
    outputs = (group_type**) malloc((data_reader.rows) * sizeof(group_type*));
    if (!outputs)
        ERROR_LOG("Outputs memory allocation error!\n");
    for (size_t i = 0; (i < data_reader.rows) ; i++)
    {
        inputs[i] = (group_type*) malloc(sizeof(group_type) * INPUTS / BATCH);
        outputs[i] = (group_type*) malloc(sizeof(group_type) * OUTPUTS / BATCH);
        for (size_t j = 0; (j < (INPUTS / BATCH)) ; j++)
        {
            inputs[i][j] = floats_to_uint32(&(data_reader.in[i][j * BATCH]));
        }
        for (size_t j = 0; (j < (OUTPUTS / BATCH)) ; j++)
        {
            outputs[i][j] = floats_part_to_uint32(
                    &(data_reader.tg[i][j * BATCH]), data_reader.nops);
        }
    }
    DESCRIBE_LOG("Data conversion ended\n");
    
    // Train, baby, train.
    DESCRIBE_LOG("Learning started\n");
    for (size_t it = 0; (it < 10) ; it++)
    {
        pop_selection(&pranger);
    }
    return 0;
}

