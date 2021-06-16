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
        net.hidden[0]->activator = activator;
        net.hidden[0]->inputs = 0;
        net.hidden[0]->weights = hidden1_weights;
#if (LAYER_II_NEURONS > 0)
        net.hidden[0]->output = hidden2;
#else
        net.hidden[0]->output = 0;
#endif
    }
#endif
#if (LAYER_II_NEURONS > 0)
    net.hidden[1] = hidden2;
    for (i = 0;i < LAYER_II_NEURONS;i++)
    {
        net.hidden[1]->activator = activator;
        net.hidden[1]->inputs = 0;
        net.hidden[1]->weights = hidden2_weights;
#if (LAYER_III_NEURONS > 0)
        net.hidden[1]->output = hidden3;
#else
        net.hidden[1]->output = 0;
#endif
    }
#endif
#if (LAYER_III_NEURONS > 0)
    net.hidden[2] = hidden3;
    for (i = 0;i < LAYER_III_NEURONS;i++)
    {
        net.hidden[2]->activator = activator;
        net.hidden[2]->inputs = 0;
        net.hidden[2]->weights = hidden3_weights;
#if (LAYER_IV_NEURONS > 0)
        net.hidden[2]->output = hidden4;
#else
        net.hidden[2]->output = 0;
#endif
    }
#endif
#if (LAYER_IV_NEURONS > 0)
    net.hidden[3] = hidden4;
    for (i = 0;i < LAYER_IV_NEURONS;i++)
    {
        net.hidden[3]->activator = activator;
        net.hidden[3]->inputs = 0;
        net.hidden[3]->weights = hidden4_weights;
#if (LAYER_V_NEURONS > 0)
        net.hidden[3]->output = hidden5;
#else
        net.hidden[3]->output = 0;
#endif
    }
#endif
#if (LAYER_V_NEURONS > 0)
    net.hidden[4] = hidden5;
    for (i = 0;i < LAYER_V_NEURONS;i++)
    {
        net.hidden[4]->activator = activator;
        net.hidden[4]->inputs = 0;
        net.hidden[4]->weights = hidden5_weights;
#if (LAYER_VI_NEURONS > 0)
        net.hidden[4]->output = hidden6;
#else
        net.hidden[4]->output = 0;
#endif
    }
#endif
#if (LAYER_VI_NEURONS > 0)
    net.hidden[5] = hidden6;
    for (i = 0;i < LAYER_VI_NEURONS;i++)
    {
        net.hidden[5]->activator = activator;
        net.hidden[5]->inputs = 0;
        net.hidden[5]->weights = hidden6_weights;
#if (LAYER_VII_NEURONS > 0)
        net.hidden[5]->output = hidden7;
#else
        net.hidden[5]->output = 0;
#endif
    }
#endif
#if (LAYER_VII_NEURONS > 0)
    net.hidden[6] = hidden7;
    for (i = 0;i < LAYER_VII_NEURONS;i++)
    {
        net.hidden[6]->activator = activator;
        net.hidden[6]->inputs = 0;
        net.hidden[6]->weights = hidden7_weights;
        net.hidden[6]->output = 0;
    }
#endif
}