#include "advanced_filter_ecg_ppg.h"
#include "data_reader.h"
#include "logger.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 2;
    const int nops = 0;
    float target[3];
    adv_filter filter;
    data data;
    
    // Load the training set.
    DESCRIBE_LOG("Read started\n");
    build_data(&data, "tests/prep_data.txt", "\t", nips, nops);
    DESCRIBE_LOG("Files readed %zu rows\n", data.rows);

    // Train, baby, train.
    DESCRIBE_LOG("Initialization started\n");
    adv_filter_init(&filter);
    DESCRIBE_LOG("Load started\n");
    DESCRIBE_LOG("Inference started\n");
    for (int k = 0; k < data.rows; k++)
    {
        adv_filter_input(&filter, (uint16_t)data.in[k][0], (uint16_t)data.in[k][1]);
    }
    return 0;
}
