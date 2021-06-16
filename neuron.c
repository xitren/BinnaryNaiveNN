#include "neuron.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if (LAYER_I_NEURONS > 0)
    static neuron hidden1[LAYER_I_NEURONS];
    static float hidden1_weights[LAYER_I_NEURONS][INPUTS];
#endif
#if (LAYER_II_NEURONS > 0)
    static neuron hidden2[LAYER_II_NEURONS];
    static float hidden2_weights[LAYER_II_NEURONS][LAYER_I_NEURONS];
#endif
#if (LAYER_III_NEURONS > 0)
    static neuron hidden3[LAYER_III_NEURONS];
    static float hidden3_weights[LAYER_III_NEURONS][LAYER_II_NEURONS];
#endif
#if (LAYER_IV_NEURONS > 0)
    static neuron hidden4[LAYER_IV_NEURONS];
    static float hidden4_weights[LAYER_IV_NEURONS][LAYER_III_NEURONS];
#endif
#if (LAYER_V_NEURONS > 0)
    static neuron hidden5[LAYER_V_NEURONS];
    static float hidden5_weights[LAYER_V_NEURONS][LAYER_IV_NEURONS];
#endif
#if (LAYER_VI_NEURONS > 0)
    static neuron hidden6[LAYER_VI_NEURONS];
    static float hidden6_weights[LAYER_VI_NEURONS][LAYER_V_NEURONS];
#endif
#if (LAYER_VII_NEURONS > 0)
    static neuron hidden7[LAYER_VII_NEURONS];
    static float hidden7_weights[LAYER_VII_NEURONS][LAYER_VI_NEURONS];
#endif
    
void nn_initialize(network net, activation_f activator)
{
    int i;
    for (i = 0;i < INPUTS;i++)
    {
        net.inputs[i] = 0;
    }
    for (i = 0;i < INPUTS;i++)
    {
        net.outputs[i] = 0;
    }
#if (LAYER_I_NEURONS > 0)
    net.hidden[0] = hidden1;
    for (i = 0;i < LAYER_I_NEURONS;i++)
    {
        net.hidden[0][i]->activator = activator;
        net.hidden[0][i]->inputs = 0;
        net.hidden[0][i]->weights = hidden1_weights[i];
#if (LAYER_II_NEURONS > 0)
        net.hidden[0][i]->output = hidden2;
#else
        net.hidden[0][i]->output = 0;
#endif
    }
#endif
#if (LAYER_II_NEURONS > 0)
    net.hidden[1] = hidden2;
    for (i = 0;i < LAYER_II_NEURONS;i++)
    {
        net.hidden[1][i]->activator = activator;
        net.hidden[1][i]->inputs = 0;
        net.hidden[1][i]->weights = hidden2_weights[i];
#if (LAYER_III_NEURONS > 0)
        net.hidden[1][i]->output = hidden3;
#else
        net.hidden[1][i]->output = 0;
#endif
    }
#endif
#if (LAYER_III_NEURONS > 0)
    net.hidden[2] = hidden3;
    for (i = 0;i < LAYER_III_NEURONS;i++)
    {
        net.hidden[2][i]->activator = activator;
        net.hidden[2][i]->inputs = 0;
        net.hidden[2][i]->weights = hidden3_weights[i];
#if (LAYER_IV_NEURONS > 0)
        net.hidden[2][i]->output = hidden4;
#else
        net.hidden[2][i]->output = 0;
#endif
    }
#endif
#if (LAYER_IV_NEURONS > 0)
    net.hidden[3] = hidden4;
    for (i = 0;i < LAYER_IV_NEURONS;i++)
    {
        net.hidden[3][i]->activator = activator;
        net.hidden[3][i]->inputs = 0;
        net.hidden[3][i]->weights = hidden4_weights[i];
#if (LAYER_V_NEURONS > 0)
        net.hidden[3][i]->output = hidden5;
#else
        net.hidden[3][i]->output = 0;
#endif
    }
#endif
#if (LAYER_V_NEURONS > 0)
    net.hidden[4] = hidden5;
    for (i = 0;i < LAYER_V_NEURONS;i++)
    {
        net.hidden[4][i]->activator = activator;
        net.hidden[4][i]->inputs = 0;
        net.hidden[4][i]->weights = hidden5_weights[i];
#if (LAYER_VI_NEURONS > 0)
        net.hidden[4][i]->output = hidden6;
#else
        net.hidden[4][i]->output = 0;
#endif
    }
#endif
#if (LAYER_VI_NEURONS > 0)
    net.hidden[5] = hidden6;
    for (i = 0;i < LAYER_VI_NEURONS;i++)
    {
        net.hidden[5][i]->activator = activator;
        net.hidden[5][i]->inputs = 0;
        net.hidden[5][i]->weights = hidden6_weights[i];
#if (LAYER_VII_NEURONS > 0)
        net.hidden[5][i]->output = hidden7;
#else
        net.hidden[5][i]->output = 0;
#endif
    }
#endif
#if (LAYER_VII_NEURONS > 0)
    net.hidden[6] = hidden7;
    for (i = 0;i < LAYER_VII_NEURONS;i++)
    {
        net.hidden[6][i]->activator = activator;
        net.hidden[6][i]->inputs = 0;
        net.hidden[6][i]->weights = hidden7_weights[i];
        net.hidden[6][i]->output = 0;
    }
#endif
}