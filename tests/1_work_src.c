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
    
    const int nips = 255;
    const int nops = 10;
    float target[10];
    data data;
    
    // Load the training set.
    DESCRIBE_LOG("Read started\n");
    build_data(&data, "tests/semeion.data", " ", nips, nops);
    DESCRIBE_LOG("Files readed\n");
    
    // Train, baby, train.
    network net;
    DESCRIBE_LOG("Initialization started\n");
    nn_initialize(&net,&activation,&pd_activation);
    net.teaching_speed = 4;
    float error = 0.9 * data.rows;
    DESCRIBE_LOG("Learning started\n");
    for (int it = 0; (it < 10000) 
            && (net.teaching_speed > 0.001) 
            && ((error / data.rows) > 0.01); it++)
    {
        error = 0.;
        for (int row = 0; row < data.rows; row++)
        {
            for (int i = 0; i < nips; i++)
            {
                net.inputs[i] = data.in[row][i];
            }
            for (int i = 0; i < OUTPUTS; i++)
            {
                target[i] = data.tg[row][i];
            }
            TRACE_LOG("Target: %f %f %f\n", target[0], target[1], target[2]);
            nn_backward(&net,target);
            TRACE_LOG("Outputs: %f %f %f\n", net.outputs[0], net.outputs[1], net.outputs[2]);
            error += toterr(target, net.outputs, OUTPUTS);
        }
        net.teaching_speed *= 0.994f;
        DESCRIBE_LOG("%d) error %.12f :: learning rate %f\n",
            it,
            (double) error / data.rows,
            (double) net.teaching_speed);
        nn_save(&net, "net.txt");
    }
    return 0;
}
