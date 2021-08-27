#include "neuron.h"

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
    
// Returns floating point random from 0.0 - 1.0.
static inline float frand();
static float nn_neuron_activation(const network *net, const neuron *one);

void nn_initialize(network *net, activation_f activator, pd_activation_f pd_activator)
{
    size_t i, j, k;
    for (i = 0;i < INPUTS;i++)
    {
        net->inputs[i] = 0;
    }
    for (i = 0;i < OUTPUTS;i++)
    {
        net->outputs[i] = 0;
    }
    net->hidden_cnt = sizer;
#if (LAYER_I_NEURONS > 0)
    net->hidden[0] = hidden1;
    for (i = 0;i < LAYER_I_NEURONS;i++)
    {
        net->hidden[0][i].ni = INPUTS;
        net->hidden[0][i].activator = activator;
        net->hidden[0][i].pd_activator = pd_activator;
        net->hidden[0][i].inputs = 0;
        net->hidden[0][i].weights = hidden1_weights[i];
#if (LAYER_II_NEURONS > 0)
        net->hidden[0][i].outputs = (neuron *)hidden2;
        net->hidden[0][i].no = LAYER_II_NEURONS;
#else
        net->hidden[0][i].outputs = 0;
        net->hidden[0][i].no = 0;
#endif
    }
#endif
#if (LAYER_II_NEURONS > 0)
    net->hidden[1] = hidden2;
    for (i = 0;i < LAYER_II_NEURONS;i++)
    {
        net->hidden[1][i].ni = LAYER_I_NEURONS;
        net->hidden[1][i].activator = activator;
        net->hidden[1][i].pd_activator = pd_activator;
        net->hidden[1][i].inputs = (neuron *)hidden1;
        net->hidden[1][i].weights = hidden2_weights[i];
#if (LAYER_III_NEURONS > 0)
        net->hidden[1][i].outputs = (neuron *)hidden3;
        net->hidden[1][i].no = LAYER_III_NEURONS;
#else
        net->hidden[1][i].outputs = 0;
        net->hidden[1][i].no = 0;
#endif
    }
#endif
#if (LAYER_III_NEURONS > 0)
    net->hidden[2] = hidden3;
    for (i = 0;i < LAYER_III_NEURONS;i++)
    {
        net->hidden[2][i].ni = LAYER_II_NEURONS;
        net->hidden[2][i].activator = activator;
        net->hidden[2][i].pd_activator = pd_activator;
        net->hidden[2][i].inputs = (neuron *)hidden2;
        net->hidden[2][i].weights = hidden3_weights[i];
#if (LAYER_IV_NEURONS > 0)
        net->hidden[2][i].outputs = (neuron *)hidden4;
        net->hidden[2][i].no = LAYER_IV_NEURONS;
#else
        net->hidden[2][i].outputs = 0;
        net->hidden[2][i].no = 0;
#endif
    }
#endif
#if (LAYER_IV_NEURONS > 0)
    net->hidden[3] = hidden4;
    for (i = 0;i < LAYER_IV_NEURONS;i++)
    {
        net->hidden[3][i].ni = LAYER_III_NEURONS;
        net->hidden[3][i].activator = activator;
        net->hidden[3][i].pd_activator = pd_activator;
        net->hidden[3][i].inputs = hidden3;
        net->hidden[3][i].weights = hidden4_weights[i];
#if (LAYER_V_NEURONS > 0)
        net->hidden[3][i].outputs = (neuron *)hidden5;
        net->hidden[3][i].no = LAYER_V_NEURONS;
#else
        net->hidden[3][i].outputs = 0;
        net->hidden[3][i].no = 0;
#endif
    }
#endif
#if (LAYER_V_NEURONS > 0)
    net->hidden[4] = hidden5;
    for (i = 0;i < LAYER_V_NEURONS;i++)
    {
        net->hidden[4][i].ni = LAYER_IV_NEURONS;
        net->hidden[4][i].activator = activator;
        net->hidden[4][i].pd_activator = pd_activator;
        net->hidden[4][i].inputs = hidden4;
        net->hidden[4][i].weights = hidden5_weights[i];
#if (LAYER_VI_NEURONS > 0)
        net->hidden[4][i].outputs = (neuron *)hidden6;
        net->hidden[4][i].no = LAYER_VI_NEURONS;
#else
        net->hidden[4][i].outputs = 0;
        net->hidden[4][i].no = 0;
#endif
    }
#endif
#if (LAYER_VI_NEURONS > 0)
    net->hidden[5] = hidden6;
    for (i = 0;i < LAYER_VI_NEURONS;i++)
    {
        net->hidden[5][i].ni = LAYER_V_NEURONS;
        net->hidden[5][i].activator = activator;
        net->hidden[5][i].pd_activator = pd_activator;
        net->hidden[5][i].inputs = hidden5;
        net->hidden[5][i].weights = hidden6_weights[i];
#if (LAYER_VII_NEURONS > 0)
        net->hidden[5][i].outputs = (neuron *)hidden7;
        net->hidden[5][i].no = LAYER_VII_NEURONS;
#else
        net->hidden[5][i].outputs = 0;
        net->hidden[5][i].no = 0;
#endif
    }
#endif
#if (LAYER_VII_NEURONS > 0)
    net->hidden[6] = hidden7;
    for (i = 0;i < LAYER_VII_NEURONS;i++)
    {
        net->hidden[6][i].ni = LAYER_VI_NEURONS;
        net->hidden[6][i].activator = activator;
        net->hidden[6][i].pd_activator = pd_activator;
        net->hidden[6][i].inputs = hidden6;
        net->hidden[6][i].weights = hidden7_weights[i];
        net->hidden[6][i].outputs = 0;
        net->hidden[6][i].no = 0;
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
    net->teaching_speed = 1.;
}

void nn_inference(network *net)
{
    size_t i,j;
    for (i = 0;i < LAYERS;i++)
    {
        neuron *line;
        line = (neuron *)net->hidden[i];
        DEBUG_PRINT("Layer %zd\n", i);
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron *one;
            one = line + j;
            DEBUG_PRINT("1\n");
            one->output = nn_neuron_activation(net, one);
            DEBUG_PRINT("Output\n");
            if (!one->outputs)
            {
                net->outputs[j] = one->output;
            }
            DEBUG_PRINT("Layer %zd, neuron %zd: out(%f) \r\n\r", i, j, one->output);
        }
    }
}

void nn_save(network *net, const char* path)
{
    size_t i,j,k;
    FILE* const file = fopen(path, "w");
    for (i = 0;i < LAYERS;i++)
    {
        fprintf(file, "===========\n");
        neuron *line;
        line = (neuron *)net->hidden[i];
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron *one;
            one = line + j;
            fprintf(file, "%f\n", one->bias);
            fprintf(file, "%ld\n", one->ni);
            for (k = 0;k < one->ni;k++)
            {
                fprintf(file, "%f\t", one->weights[k]);
            }
            fprintf(file, "\n");
        }
    }
    fclose(file);
}

void nn_load(network *net, const char* path)
{
    size_t i,j,k;
    FILE* const file = fopen(path, "r");
    for (i = 0;i < LAYERS;i++)
    {
        fscanf(file, "===========\n");
        neuron *line;
        line = (neuron *)net->hidden[i];
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron *one;
            one = line + j;
            fscanf(file, "%f\n", &(one->bias));
            fscanf(file, "%zu\n", &(one->ni));
            for (k = 0;k < one->ni;k++)
            {
                fscanf(file, "%f\t", &(one->weights[k]));
            }
            fscanf(file, "\n");
        }
    }
}

static float nn_neuron_activation(const network *net, const neuron *one)
{
    size_t k;
    float sum;
    sum = one->bias;
    if (one->inputs)
    {
        for (k = 0;k < one->ni;k++)
        {
            sum += ((neuron *)(one->inputs))[k].output * one->weights[k];
        }
    }
    else
    {
        for (k = 0;k < INPUTS;k++)
        {
            sum += net->inputs[k] * one->weights[k];
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
                    DEBUG_PRINT("Sum %f %f\r\n\r", one->delta, gg->weights[j]);
                }
                one->delta = pd * one->delta;
            }
            else
            {
                one->delta = pd * (one->output - target[j]);
            }
            DEBUG_PRINT("Layer %ld, neuron %ld: d(%f) \r\n\r", i - 1, j, one->delta);
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
                    DEBUG_PRINT("Layer %ld, neuron %ld, link %ld: w(%f) \r\n\r", 
                            i, j, k, one->weights[k]);
                }
            }
            else
            {
                for (k = 0;k < one->ni;k++)
                {
                    one->weights[k] -= net->teaching_speed * one->delta
                            * net->inputs[k];
                    DEBUG_PRINT("Layer %ld, neuron %ld, link %ld: w(%f) \r\n\r", 
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

// Returns floating point random from -1.0 - 1.0.
static inline float frand()
{
    return (rand() / (float) (RAND_MAX / 2)) - 1.0;
}