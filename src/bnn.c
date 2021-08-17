
#include "bnn.h"

#define GET_BIT(num, n) ((num >> n) & 1)
#define PTR_CAST(ptr) ((neuron_batch *)(ptr))
#define PTR_UNCAST(ptr) ((void *)(ptr))
#define POW2(val) (val * val)

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
static void nn_neuron_batch_activation_full(const network *net, const neuron_batch *one);

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
        net->hidden[0][i].outputs = PTR_UNCAST(hidden2);
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
        net->hidden[1][i].inputs = PTR_UNCAST(hidden1);
        net->hidden[1][i].weights = hidden2_weights[i];
#if (LAYER_III_NEURONS > 0)
        net->hidden[1][i].outputs = PTR_UNCAST(hidden3);
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
        net->hidden[2][i].inputs = PTR_UNCAST(hidden2);
        net->hidden[2][i].weights = hidden3_weights[i];
#if (LAYER_IV_NEURONS > 0)
        net->hidden[2][i].outputs = PTR_UNCAST(hidden4);
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
        net->hidden[3][i].inputs = PTR_UNCAST(hidden3);
        net->hidden[3][i].weights = hidden4_weights[i];
#if (LAYER_V_NEURONS > 0)
        net->hidden[3][i].outputs = PTR_UNCAST(hidden5);
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
        net->hidden[4][i].inputs = PTR_UNCAST(hidden4);
        net->hidden[4][i].weights = hidden5_weights[i];
#if (LAYER_VI_NEURONS > 0)
        net->hidden[4][i].outputs = PTR_UNCAST(hidden6);
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
        net->hidden[5][i].inputs = PTR_UNCAST(hidden5);
        net->hidden[5][i].weights = hidden6_weights[i];
#if (LAYER_VII_NEURONS > 0)
        net->hidden[5][i].outputs = PTR_UNCAST(hidden7);
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
        net->hidden[6][i].inputs = PTR_UNCAST(hidden6);
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

#ifdef LEARNER
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
            one->output = nn_neuron_batch_activation_full(net, one);
            if (!one->outputs)
            {
                net->outputs[j] = one->output;
            }
            printf("Layer %d, neuron %d: out(%f) \r\n\r", i, j, one->output);
        }
    }
}

static double nn_neuron_batch_activation_full(const network *net, const neuron_batch *one)
{
    size_t i,k,j;
    double sum, w_tanh;
    for (i = 0;i < BATCH;i++)
    {
        sum = tanh(one->bias_full[i]);
        if (one->inputs)
        {
            for (k = 0;k < one->ni;k++)
            {
                for (j = 0;j < BATCH;j++)
                {
                    w_tanh = tanh(one->weights_full[k * BATCH + j]);
                    sum += PTR_CAST(one->inputs)[k].output_full[j] * w_tanh;
                }
            }
        }
        else
        {
            for (k = 0;k < INPUTS;k++)
            {
                for (j = 0;j < BATCH;j++)
                {
                    w_tanh = tanh(one->weights_full[k * BATCH + j]);
                    sum += GET_BIT(net->inputs[k], j) * w_tanh;
                }
            }
        }
        one->output_a_full[i] = sum;
        one->output_full[i] = tanh(sum);
    }
}

void nn_backward(network *net, GROUP_TYPE target[OUTPUTS / BATCH])
{
    size_t i,j,k,b,b2;
    double sum,bit;
    nn_inference_learning(net);
    // Delta propagation
    for (i = LAYERS;i > 0;i--)
    {
        neuron_batch *line;
        line = net->hidden[i - 1];
        for (j = 0;j < net->hidden_cnt[i - 1];j++)
        {
            neuron_batch *one;
            one = line + j;
            if (one->outputs)
            {
                for (b = 0;b < BATCH;b++)
                {
                    one->delta[b] = 0.0;
                    for (k = 0;k < one->no;k++)
                    {
                        neuron_batch *gg = ((neuron_batch *)(one->outputs)) + k;
                        for (b2 = 0;b2 < BATCH;b2++)
                        {
                            one->delta[b] += tanh(gg->weights_full[j * BATCH + b]) * gg->delta[b2];
                        }
                    }
                    one->delta[b] *= (1 - POW2(tanh(one->output_a_full[b])));
                }
            }
            else
            {
                for (b = 0;b < BATCH;b++)
                {
                    bit = GET_BIT(target, j*BATCH + b);
                    one->delta[b] = (one->output_full[b] - bit);
                    one->delta[b] *= (1 - POW2(tanh(one->output_a_full[b])));
                }
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
#endif

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