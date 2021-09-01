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
    nn_set_weights(chr->genes);
    nn_set_beta(&bnn, &(chr->genes[(CHROMOSOME_SIZE / BATCH) - 1 - BATCHES_SIZE]));
    err = nn_error(&bnn, inputs, outputs, data_reader.rows);
    return err;
}

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    DESCRIBE_LOG("main\n");
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 1400;
    const int nops = 1400;
    population_ranger pranger;
    chromosome_binary best[POP_MAX];
    chromosome_binary winner;
    
    // Load the training set.
    DESCRIBE_LOG("Read started\n");
    build_data(&data_reader, "tests/input_data_bnn_1400-1400.txt", " ", nips, nops);
    DESCRIBE_LOG("Files readed\n");
    
    DESCRIBE_LOG("Initialization started\n");
    nn_initialize(&bnn);
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
        outputs[i][0] = floats_part_to_uint32(data_reader.tg[i], data_reader.nops);
    }
    DESCRIBE_LOG("Data conversion ended\n");
    
    // Train, baby, train.
    DESCRIBE_LOG("Learning started\n");
    for (size_t runs = 0;runs < POP_MAX; runs++)
    {
        DESCRIBE_LOG("%zu population started\n", runs);
        float min = 10000000;
        pranger.err_calc[0] = min;
        initiate_population_ranger(&pranger, &error_parser, POP_MAX, 0.5, 0.5);
        for (size_t it = 0; (it < 1000) && (pranger.err_calc[0] > 0.01) ; it++)
        {
            pop_selection(&pranger);
            if (!(it % 100))
                DESCRIBE_LOG("Iteration %zu\n", it);
            if (pranger.err_calc[0] < min)
            {
                min = pranger.err_calc[0];
                best[runs] = pranger.pop_live[0];
            }
        }
    }
    float min = 10000000;
    pranger.err_calc[0] = min;
    DESCRIBE_LOG("Best population started\n");
    initiate_population(&pranger, best, &error_parser, POP_MAX, 0.1, 0.5);
    for (size_t it = 0; (it < 1000) && (pranger.err_calc[0] > 0.01) ; it++)
    {
        pop_selection(&pranger);
        if (!(it % 100))
            DESCRIBE_LOG("Iteration %zu\n", it);
        if (pranger.err_calc[0] < min)
        {
            min = pranger.err_calc[0];
            winner = pranger.pop_live[0];
        }
    }
    DESCRIBE_LOG("The winner is %f\n", min);
    nn_set_weights(winner.genes);
    nn_set_beta(&bnn, &(winner.genes[(CHROMOSOME_SIZE / BATCH) - 1 - 3]));
    nn_save(&bnn, "bnn_CIFAR.net");
    return 0;
}

