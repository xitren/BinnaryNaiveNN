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
    net.teaching_speed = 4;
    float error = 0.9 * data_result.rows;
    DESCRIBE_LOG("Learning started\n");
    for (int it = 0; (it < 10000) 
            && (net.teaching_speed > 0.001) 
            && ((error / data_result.rows) > 0.00001); it++)
    {
        error = 0.;
        for (int row = 0; row < data_result.rows; row++)
        {
            for (int i = 0; i < nips; i++)
            {
                if (i < 100)
                    net.inputs[i] = data_ecg.in[row][i];
                else
                    net.inputs[i] = data_ppg.in[row][i - 100];
            }
            for (int i = 0; i < OUTPUTS; i++)
            {
                target[i] = data_result.tg[row][i];
            }
//            TRACE_LOG("Target: %f %f %f\n", target[0], target[1], target[2]);
            nn_backward(&net,target);
//            TRACE_LOG("Outputs: %f %f %f\n", net.outputs[0], net.outputs[1], net.outputs[2]);
            error += toterr(target, net.outputs, OUTPUTS);
        }
        net.teaching_speed *= 0.9f;
        DESCRIBE_LOG("%d) error %.12f :: learning rate %f\n",
            it,
            (double) error / data_result.rows,
            (double) net.teaching_speed);
        nn_save(&net, "work_real_135-135-1.net");
    }
    char fname[100];
    snprintf(fname, sizeof(fname), "work_real_135-135-1-%f.net", error / data_result.rows);
    nn_save(&net, fname);
    return 0;
}
