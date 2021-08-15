
#include "bnn.h"

#if (LAYER_I_NEURONS > 0)
    static neuron_batch hidden1[LAYER_I_NEURONS / BATCH];
    static GROUP_TYPE hidden1_weights[LAYER_I_NEURONS / BATCH][INPUTS];
    #ifdef LEARNER
        static double hidden1_weights_full[LAYER_I_NEURONS / BATCH][INPUTS * BATCH];
    #endif
#endif
#if (LAYER_II_NEURONS > 0)
    static neuron_batch hidden2[LAYER_II_NEURONS / BATCH];
    static GROUP_TYPE hidden2_weights[LAYER_II_NEURONS / BATCH][LAYER_I_NEURONS];
    #ifdef LEARNER
        static double hidden2_weights_full[LAYER_II_NEURONS / BATCH][LAYER_I_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_III_NEURONS > 0)
    static neuron_batch hidden3[LAYER_III_NEURONS / BATCH];
    static GROUP_TYPE hidden3_weights[LAYER_III_NEURONS / BATCH][LAYER_II_NEURONS];
    #ifdef LEARNER
        static double hidden3_weights_full[LAYER_III_NEURONS / BATCH][LAYER_II_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_IV_NEURONS > 0)
    static neuron_batch hidden4[LAYER_IV_NEURONS / BATCH];
    static GROUP_TYPE hidden4_weights[LAYER_IV_NEURONS / BATCH][LAYER_III_NEURONS];
    #ifdef LEARNER
        static double hidden4_weights_full[LAYER_IV_NEURONS / BATCH][LAYER_III_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_V_NEURONS > 0)
    static neuron_batch hidden5[LAYER_V_NEURONS / BATCH];
    static GROUP_TYPE hidden5_weights[LAYER_V_NEURONS / BATCH][LAYER_IV_NEURONS];
    #ifdef LEARNER
        static double hidden5_weights_full[LAYER_V_NEURONS / BATCH][LAYER_IV_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_VI_NEURONS > 0)
    static neuron_batch hidden6[LAYER_VI_NEURONS / BATCH];
    static GROUP_TYPE hidden6_weights[LAYER_VI_NEURONS / BATCH][LAYER_V_NEURONS];
    #ifdef LEARNER
        static double hidden6_weights_full[LAYER_VI_NEURONS / BATCH][LAYER_V_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_VII_NEURONS > 0)
    static neuron_batch hidden7[LAYER_VII_NEURONS / BATCH];
    static GROUP_TYPE hidden7_weights[LAYER_VII_NEURONS / BATCH][LAYER_VI_NEURONS];
    #ifdef LEARNER
        static double hidden7_weights_full[LAYER_VII_NEURONS / BATCH][LAYER_VI_NEURONS * BATCH];
    #endif
#endif
    size_t sizer[] = {
#if (LAYERS > 0)
        LAYER_I_NEURONS
#endif
#if (LAYERS > 1)
        , LAYER_II_NEURONS
#endif
#if (LAYERS > 2)
        , LAYER_III_NEURONS
#endif
#if (LAYERS > 3)
        , LAYER_IV_NEURONS
#endif
#if (LAYERS > 4)
        , LAYER_V_NEURONS
#endif
#if (LAYERS > 5)
        , LAYER_VI_NEURONS
#endif
#if (LAYERS > 6)
        , LAYER_VII_NEURONS
#endif
    };

static inline float frand();
static float nn_neuron_activation(const network *net, const neuron *one);

void nn_initialize(network *net)
{
    size_t i, j, k;
    for (i = 0;i < (INPUTS / BATCH);i++)
    {
        net->inputs[i] = 0;
    }
    for (i = 0;i < (OUTPUTS / BATCH);i++)
    {
        net->outputs[i] = 0;
    }
    net->hidden_cnt = sizer;
#if (LAYER_I_NEURONS > 0)
    net->hidden[0] = hidden1;
    for (i = 0;i < (LAYER_I_NEURONS / BATCH);i++)
    {
        net->hidden[0][i].ni = INPUTS;
        net->hidden[0][i].inputs = 0;
        net->hidden[0][i].weights = hidden1_weights[i];
#if (LAYER_II_NEURONS > 0)
        net->hidden[0][i].outputs = (neuron *)hidden2;
        net->hidden[0][i].no = LAYER_II_NEURONS;
#else
        net->hidden[0][i].outputs = 0;
        net->hidden[0][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[0][i].weights_full = hidden1_weights_full[i];
        net->hidden[0][i].bias_full = hidden1_weights_full[i];
        net->hidden[0][i].output_full = hidden1_weights_full[i];
#endif
    }
#endif
#if (LAYER_II_NEURONS > 0)
    net->hidden[1] = hidden2;
    for (i = 0;i < (LAYER_II_NEURONS / BATCH);i++)
    {
        net->hidden[1][i].ni = LAYER_I_NEURONS;
        net->hidden[1][i].inputs = (neuron *)hidden1;
        net->hidden[1][i].weights = hidden2_weights[i];
#if (LAYER_III_NEURONS > 0)
        net->hidden[1][i].outputs = (neuron *)hidden3;
        net->hidden[1][i].no = LAYER_III_NEURONS;
#else
        net->hidden[1][i].outputs = 0;
        net->hidden[1][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[1][i].weights_full = hidden2_weights_full[i];
        net->hidden[1][i].bias_full = hidden2_weights_full[i];
        net->hidden[1][i].output_full = hidden2_weights_full[i];
#endif
    }
#endif
#if (LAYER_III_NEURONS > 0)
    net->hidden[2] = hidden3;
    for (i = 0;i < (LAYER_III_NEURONS / BATCH);i++)
    {
        net->hidden[2][i].ni = LAYER_II_NEURONS;
        net->hidden[2][i].inputs = (neuron *)hidden2;
        net->hidden[2][i].weights = hidden3_weights[i];
#if (LAYER_IV_NEURONS > 0)
        net->hidden[2][i].outputs = (neuron *)hidden4;
        net->hidden[2][i].no = LAYER_IV_NEURONS;
#else
        net->hidden[2][i].outputs = 0;
        net->hidden[2][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[2][i].weights_full = hidden3_weights_full[i];
        net->hidden[2][i].bias_full = hidden3_weights_full[i];
        net->hidden[2][i].output_full = hidden3_weights_full[i];
#endif
    }
#endif
#if (LAYER_IV_NEURONS > 0)
    net->hidden[3] = hidden4;
    for (i = 0;i < (LAYER_IV_NEURONS / BATCH);i++)
    {
        net->hidden[3][i].ni = LAYER_III_NEURONS;
        net->hidden[3][i].inputs = hidden3;
        net->hidden[3][i].weights = hidden4_weights[i];
#if (LAYER_V_NEURONS > 0)
        net->hidden[3][i].outputs = (neuron *)hidden5;
        net->hidden[3][i].no = LAYER_V_NEURONS;
#else
        net->hidden[3][i].outputs = 0;
        net->hidden[3][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[3][i].weights_full = hidden4_weights_full[i];
        net->hidden[3][i].bias_full = hidden4_weights_full[i];
        net->hidden[3][i].output_full = hidden4_weights_full[i];
#endif
    }
#endif
#if (LAYER_V_NEURONS > 0)
    net->hidden[4] = hidden5;
    for (i = 0;i < (LAYER_V_NEURONS / BATCH);i++)
    {
        net->hidden[4][i].ni = LAYER_IV_NEURONS;
        net->hidden[4][i].inputs = hidden4;
        net->hidden[4][i].weights = hidden5_weights[i];
#if (LAYER_VI_NEURONS > 0)
        net->hidden[4][i].outputs = (neuron *)hidden6;
        net->hidden[4][i].no = LAYER_VI_NEURONS;
#else
        net->hidden[4][i].outputs = 0;
        net->hidden[4][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[4][i].weights_full = hidden5_weights_full[i];
        net->hidden[4][i].bias_full = hidden5_weights_full[i];
        net->hidden[4][i].output_full = hidden5_weights_full[i];
#endif
    }
#endif
#if (LAYER_VI_NEURONS > 0)
    net->hidden[5] = hidden6;
    for (i = 0;i < (LAYER_VI_NEURONS / BATCH);i++)
    {
        net->hidden[5][i].ni = LAYER_V_NEURONS;
        net->hidden[5][i].inputs = hidden5;
        net->hidden[5][i].weights = hidden6_weights[i];
#if (LAYER_VII_NEURONS > 0)
        net->hidden[5][i].outputs = (neuron *)hidden7;
        net->hidden[5][i].no = LAYER_VII_NEURONS;
#else
        net->hidden[5][i].outputs = 0;
        net->hidden[5][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[5][i].weights_full = hidden6_weights_full[i];
        net->hidden[5][i].bias_full = hidden6_weights_full[i];
        net->hidden[5][i].output_full = hidden6_weights_full[i];
#endif
    }
#endif
#if (LAYER_VII_NEURONS > 0)
    net->hidden[6] = hidden7;
    for (i = 0;i < (LAYER_VII_NEURONS / BATCH);i++)
    {
        net->hidden[6][i].ni = LAYER_VI_NEURONS;
        net->hidden[6][i].inputs = hidden6;
        net->hidden[6][i].weights = hidden7_weights[i];
        net->hidden[6][i].outputs = 0;
        net->hidden[6][i].no = 0;
#ifdef LEARNER
        net->hidden[6][i].weights_full = hidden7_weights_full[i];
        net->hidden[6][i].bias_full = hidden7_weights_full[i];
        net->hidden[6][i].output_full = hidden7_weights_full[i];
#endif
    }
#endif
    for (i = 0;i < LAYERS;i++)
    {
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            for (k = 0;k < net->hidden[i][j].ni;k++)
            {
                net->hidden[i][j].weights[k] = frand();
            }
            net->hidden[i][j].bias = 0.5;
        }
    }
    net->teaching_speed = 1;
}

static void nn_inference_learning(network *net)
{
    size_t i,j;
    for (i = 0;i < LAYERS;i++)
    {
        neuron_batch *line;
        line = (neuron_batch *)net->hidden[i];
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron_batch *one;
            one = line + j;
            one->output = nn_neuron_activation_full(net, one);
            if (!one->outputs)
            {
                net->outputs[j] = one->output;
            }
            printf("Layer %d, neuron %d: out(%f) \r\n\r", i, j, one->output);
        }
    }
}

static float nn_neuron_activation_full(const network *net, const neuron_batch *one)
{
    size_t i,k,j;
    float sum;
    for (i = 0;i < BATCH;i++)
    {
        sum = one->bias_full[i];
        if (one->inputs)
        {
            for (k = 0;k < one->ni;k++)
            {
                for (j = 0;j < BATCH;j++)
                {
                    sum += ((neuron_batch *)(one->inputs))
                            [k].output_full[j] * one->weights_full[k * BATCH + j];
                }
            }
        }
        else
        {
            for (k = 0;k < INPUTS;k++)
            {
                sum += net->inputs[k] * one->weights[k];
            }
        }
    }
    return activation(sum);
}

void nn_backward(network *net, float target[OUTPUTS])
{
    size_t i,j,k;
    float pd;
    nn_inference(net);
    // Delta propagation
    for (i = LAYERS;i > 0;i--)
    {
        neuron *line;
        line = net->hidden[i - 1];
        for (j = 0;j < net->hidden_cnt[i - 1];j++)
        {
            neuron *one;
            one = line + j;
            pd = pd_activation(one->output);
            if (one->outputs)
            {
                one->delta = 0.0f;
                for (k = 0;k < one->no;k++)
                {
                    neuron *gg = ((neuron *)(one->outputs)) + k;
                    one->delta += gg->delta * gg->weights[j];
                    printf("Sum %f %f\r\n\r", one->delta, gg->weights[j]);
                }
                one->delta = pd * one->delta;
            }
            else
            {
                one->delta = pd * (one->output - target[j]);
            }
            printf("Layer %d, neuron %d: d(%f) \r\n\r", i - 1, j, one->delta);
        }
    }
    // Weights propagation
    for (i = 0;i < LAYERS;i++)
    {
        neuron *line;
        line = net->hidden[i];
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron *one;
            one = line + j;
            if (one->inputs)
            {
                for (k = 0;k < one->ni;k++)
                {
                    one->weights[k] -= net->teaching_speed * one->delta
                            * ((neuron *)(one->inputs))[k].output;
                    printf("Layer %d, neuron %d, link %d: w(%f) \r\n\r", 
                            i, j, k, one->weights[k]);
                }
            }
            else
            {
                for (k = 0;k < one->ni;k++)
                {
                    one->weights[k] -= net->teaching_speed * one->delta
                            * net->inputs[k];
                    printf("Layer %d, neuron %d, link %d: w(%f) \r\n\r", 
                            i, j, k, one->weights[k]);
                }
            }
        }
    }
}

// Activation function.
float activation(const float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Returns partial derivative of activation function.
float pd_activation(const float a)
{
    return a * (1.0f - a);
}

// Returns floating point random from 0.0 - 1.0.
static inline float frand()
{
    return (rand() / (float) (RAND_MAX / 2)) - 1.0;
}