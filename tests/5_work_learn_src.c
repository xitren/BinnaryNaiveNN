#include "neuron.h"
#include "logger.h"
#include "data_reader.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Computes error.
static float err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Computes total error of target to output.
static float toterr(const float* const tg, const float* const o, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += err(tg[i], o[i]);
    return sum;
}

static void result_save(const data *data, const char* path)
{
    int i,j;
    FILE* const file = fopen(path, "w");
    for (i = 0;i < data->rows;i++)
    {
        for (j = 0;j < data->nops;j++)
        {
            fprintf(file, "%f\t", data->tg[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 135;
    const int nops = 1;
    const int limit_rows = 1018780;
    float target[1];
    data data_ecg;
    data data_ppg;
    data data_result;
    
    // Load the training set.
    DESCRIBE_LOG("Read started\n");
    build_data_limit(&data_ecg, "tests/input_real_data_ecg_100.txt", "\t", 100, 0, limit_rows);
    build_data_limit(&data_ppg, "tests/input_real_data_ppg_35.txt", "\t", 35, 0, limit_rows);
    build_data_limit(&data_result, "tests/input_real_data_result_1.txt", "\t", 0, 1, limit_rows);
    DESCRIBE_LOG("Files readed\n");
    
    // Train, baby, train.
    network net;
    DESCRIBE_LOG("Initialization started\n");
    nn_initialize(&net,&activation,&pd_activation);
    nn_load(&net, "work_real_135-35-4-1.net");
    DESCRIBE_LOG("Inference started\n");
    for (int row = 0; row < data_result.rows; row++)
    {
        for (int i = 0; i < nips; i++)
        {
            if (i < 100)
                net.inputs[i] = data_ecg.in[row][i];
            else
                net.inputs[i] = data_ppg.in[row][i - 100];
        }
        nn_inference(&net);
        for (int i = 0; i < nops; i++)
        {
            data_result.tg[row][i] = net.outputs[i];
        }
    }
    result_save(&data_result, "results_real_135-35-1.csv");
    return 0;
}
