
#include "bnn.h"
#include "logger.h"
#include "binary_tools.h"

#define PTR_CAST(ptr) ((neuron_batch *)(ptr))
#define PTR_UNCAST(ptr) ((void *)(ptr))
#define POW2(val) (val * val)

typedef union _tag_caster {
   group_type gt;
   uint8_t c[sizeof(group_type)];
} caster;  

const uint8_t bit_cnt[] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 
    2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 
    4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 
    2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 
    3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 
    4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 
    3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 
    4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 
    4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 
    4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 
    6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 
    3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 
    4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 
    5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 
    6, 7, 6, 7, 7, 8};
#if (LAYER_I_NEURONS > 0)
    static neuron_batch hidden1[LAYER_I_NEURONS / BATCH];
    static group_type hidden1_weights[LAYER_I_NEURONS / BATCH][INPUTS];
    static group_type hidden1_output[LAYER_I_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden1_weights_full[LAYER_I_NEURONS / BATCH][INPUTS * BATCH];
    #endif
#endif
#if (LAYER_II_NEURONS > 0)
    static neuron_batch hidden2[LAYER_II_NEURONS / BATCH];
    static group_type hidden2_weights[LAYER_II_NEURONS / BATCH][LAYER_I_NEURONS];
    static group_type hidden2_output[LAYER_II_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden2_weights_full[LAYER_II_NEURONS / BATCH][LAYER_I_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_III_NEURONS > 0)
    static neuron_batch hidden3[LAYER_III_NEURONS / BATCH];
    static group_type hidden3_weights[LAYER_III_NEURONS / BATCH][LAYER_II_NEURONS];
    static group_type hidden3_output[LAYER_II_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden3_weights_full[LAYER_III_NEURONS / BATCH][LAYER_II_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_IV_NEURONS > 0)
    static neuron_batch hidden4[LAYER_IV_NEURONS / BATCH];
    static group_type hidden4_weights[LAYER_IV_NEURONS / BATCH][LAYER_III_NEURONS];
    static group_type hidden4_output[LAYER_III_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden4_weights_full[LAYER_IV_NEURONS / BATCH][LAYER_III_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_V_NEURONS > 0)
    static neuron_batch hidden5[LAYER_V_NEURONS / BATCH];
    static group_type hidden5_weights[LAYER_V_NEURONS / BATCH][LAYER_IV_NEURONS];
    static group_type hidden5_output[LAYER_IV_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden5_weights_full[LAYER_V_NEURONS / BATCH][LAYER_IV_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_VI_NEURONS > 0)
    static neuron_batch hidden6[LAYER_VI_NEURONS / BATCH];
    static group_type hidden6_weights[LAYER_VI_NEURONS / BATCH][LAYER_V_NEURONS];
    static group_type hidden6_output[LAYER_V_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden6_weights_full[LAYER_VI_NEURONS / BATCH][LAYER_V_NEURONS * BATCH];
    #endif
#endif
#if (LAYER_VII_NEURONS > 0)
    static neuron_batch hidden7[LAYER_VII_NEURONS / BATCH];
    static group_type hidden7_weights[LAYER_VII_NEURONS / BATCH][LAYER_VI_NEURONS];
    static group_type hidden7_output[LAYER_VI_NEURONS / BATCH];
    #ifdef LEARNER
        static double hidden7_weights_full[LAYER_VII_NEURONS / BATCH][LAYER_VI_NEURONS * BATCH];
    #endif
#endif
    size_t sizer[] = {
#if (LAYERS > 0)
        LAYER_I_NEURONS / BATCH
#endif
#if (LAYERS > 1)
        , LAYER_II_NEURONS / BATCH
#endif
#if (LAYERS > 2)
        , LAYER_III_NEURONS / BATCH
#endif
#if (LAYERS > 3)
        , LAYER_IV_NEURONS / BATCH
#endif
#if (LAYERS > 4)
        , LAYER_V_NEURONS / BATCH
#endif
#if (LAYERS > 5)
        , LAYER_VI_NEURONS / BATCH
#endif
#if (LAYERS > 6)
        , LAYER_VII_NEURONS / BATCH
#endif
    };

static inline float frand();
static inline group_type gtrand();
static inline size_t nn_activation_batch(group_type *in, group_type *w, size_t n);
#ifdef LEARNER
static void nn_neuron_batch_activation_full(const network *net, neuron_batch *one);
inline static void backward_batch_delta(neuron_batch *one, size_t j);
inline static void backward_batch_delta_last(neuron_batch *one, size_t j,
        group_type target[OUTPUTS / BATCH]);
inline static void backward_batch_weight(neuron_batch *one, double n);
inline static void backward_batch_weight_first(network *net, neuron_batch *one, double n);
inline static void backward_delta(network *net, group_type target[OUTPUTS / BATCH]);
inline static void backward_weight(network *net);
#endif

void nn_initialize(network *net)
{
    size_t i, j, k, l;
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
        net->hidden[0][i].ni = INPUTS / BATCH;
        net->hidden[0][i].input_data = net->inputs;
        net->hidden[0][i].weights = (group_type*) hidden1_weights[i];
#if (LAYER_II_NEURONS > 0)
        net->hidden[0][i].output = hidden1_output + i;
        net->hidden[0][i].no = LAYER_II_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[0][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[0][i].inputs = 0;
        net->hidden[0][i].weights_full = hidden1_weights_full[i];
#if (LAYER_II_NEURONS > 0)
        net->hidden[0][i].outputs = PTR_UNCAST(hidden2);
#else
        net->hidden[0][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_II_NEURONS > 0)
    net->hidden[1] = hidden2;
    for (i = 0;i < (LAYER_II_NEURONS / BATCH);i++)
    {
        net->hidden[1][i].ni = LAYER_I_NEURONS / BATCH;
        net->hidden[1][i].input_data = hidden1_output;
        net->hidden[1][i].weights = (group_type*) hidden2_weights[i];
#if (LAYER_III_NEURONS > 0)
        net->hidden[1][i].output = hidden2_output + i;
        net->hidden[1][i].no = LAYER_III_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[1][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[1][i].weights_full = hidden2_weights_full[i];
        net->hidden[1][i].inputs = PTR_UNCAST(hidden1);
#if (LAYER_III_NEURONS > 0)
        net->hidden[1][i].outputs = PTR_UNCAST(hidden3);
#else
        net->hidden[1][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_III_NEURONS > 0)
    net->hidden[2] = hidden3;
    for (i = 0;i < (LAYER_III_NEURONS / BATCH);i++)
    {
        net->hidden[2][i].ni = LAYER_II_NEURONS / BATCH;
        net->hidden[2][i].input_data = hidden2_output;
        net->hidden[2][i].weights = (group_type*) hidden3_weights[i];
#if (LAYER_IV_NEURONS > 0)
        net->hidden[2][i].output = hidden3_output + i;
        net->hidden[2][i].no = LAYER_IV_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[2][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[2][i].weights_full = hidden3_weights_full[i];
        net->hidden[2][i].inputs = PTR_UNCAST(hidden2);
#if (LAYER_IV_NEURONS > 0)
        net->hidden[2][i].outputs = PTR_UNCAST(hidden4);
#else
        net->hidden[2][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_IV_NEURONS > 0)
    net->hidden[3] = hidden4;
    for (i = 0;i < (LAYER_IV_NEURONS / BATCH);i++)
    {
        net->hidden[3][i].ni = LAYER_III_NEURONS / BATCH;
        net->hidden[3][i].input_data = hidden3_output;
        net->hidden[3][i].weights = (group_type*) hidden4_weights[i];
#if (LAYER_V_NEURONS > 0)
        net->hidden[3][i].output = hidden4_output + i;
        net->hidden[3][i].no = LAYER_V_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[3][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[3][i].weights_full = hidden4_weights_full[i];
        net->hidden[3][i].inputs = PTR_UNCAST(hidden3);
#if (LAYER_V_NEURONS > 0)
        net->hidden[3][i].outputs = PTR_UNCAST(hidden5);
#else
        net->hidden[3][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_V_NEURONS > 0)
    net->hidden[4] = hidden5;
    for (i = 0;i < (LAYER_V_NEURONS / BATCH);i++)
    {
        net->hidden[4][i].ni = LAYER_IV_NEURONS / BATCH;
        net->hidden[4][i].input_data = hidden4_output;
        net->hidden[4][i].weights = (group_type*) hidden5_weights[i];
#if (LAYER_VI_NEURONS > 0)
        net->hidden[4][i].output = hidden5_output + i;
        net->hidden[4][i].no = LAYER_VI_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[4][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[4][i].weights_full = hidden5_weights_full[i];
        net->hidden[4][i].inputs = PTR_UNCAST(hidden4);
#if (LAYER_VI_NEURONS > 0)
        net->hidden[4][i].outputs = PTR_UNCAST(hidden6);
#else
        net->hidden[4][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_VI_NEURONS > 0)
    net->hidden[5] = hidden6;
    for (i = 0;i < (LAYER_VI_NEURONS / BATCH);i++)
    {
        net->hidden[5][i].ni = LAYER_V_NEURONS / BATCH;
        net->hidden[5][i].input_data = hidden5_output;
        net->hidden[5][i].weights = (group_type*) hidden6_weights[i];
#if (LAYER_VII_NEURONS > 0)
        net->hidden[5][i].output = hidden6_output + i;
        net->hidden[5][i].no = LAYER_VII_NEURONS / BATCH;
#else
        net->hidden[0][i].output = net->outputs + i;
        net->hidden[5][i].no = 0;
#endif
#ifdef LEARNER
        net->hidden[5][i].weights_full = hidden6_weights_full[i];
        net->hidden[5][i].inputs = PTR_UNCAST(hidden5);
#if (LAYER_VII_NEURONS > 0)
        net->hidden[5][i].outputs = PTR_UNCAST(hidden7);
#else
        net->hidden[5][i].outputs = 0;
#endif
#endif
    }
#endif
#if (LAYER_VII_NEURONS > 0)
    net->hidden[6] = hidden7;
    for (i = 0;i < (LAYER_VII_NEURONS / BATCH);i++)
    {
        net->hidden[6][i].ni = LAYER_VI_NEURONS / BATCH;
        net->hidden[6][i].input_data = hidden6_output;
        net->hidden[6][i].weights = (group_type*) hidden7_weights[i];
        net->hidden[6][i].output = net->outputs + i;
        net->hidden[6][i].no = 0;
#ifdef LEARNER
        net->hidden[6][i].weights_full = hidden7_weights_full[i];
        net->hidden[6][i].inputs = PTR_UNCAST(hidden6);
        net->hidden[6][i].outputs = 0;
#endif
    }
#endif
    for (i = 0;i < LAYERS;i++)
    {
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            for (l = 0;l < BATCH;l++)
            {
                for (k = 0;k < net->hidden[i][j].ni;k++)
                {
                    net->hidden[i][j].weights[l * (net->hidden_cnt[i]) + k] = gtrand();
#ifdef LEARNER
                    net->hidden[i][j].weights_full[l * (net->hidden_cnt[i]) + k] = frand();
#endif
                }
            }
#ifdef LEARNER
            for (k = 0;k < BATCH;k++)
            {
                net->hidden[i][j].bias = 0;
                net->hidden[i][j].bias_full[k] = 0.5;
                net->hidden[i][j].output_full[k] = 0.0;
            }
#endif
            net->hidden[i][j]->beta = 8;
        }
    }
    net->teaching_speed = 1;
}


void nn_inference(network *net)
{
    size_t i,j;
    for (i = 0;i < LAYERS;i++)
    {
        neuron_batch *line;
        line = (neuron_batch *)net->hidden[i];
        TRACE_LOG("Layer %zu\n", i);
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            neuron_batch *one;
            one = line + j;
            TRACE_LOG("Layer %zu, neuron batch %zu \n", i, j);
            nn_activation_batch(one);
        }
    }
}

static inline void nn_activation_batch(neuron_batch *one)
{
    caster cast;
    group_type res;
    group_type ret;
    size_t i,j,k;
    ret = 0;
    for (j = 0;j < BATCH;j++)
    {
        k = 0;
        for (i = 0;i < one->ni;i++)
        {
            res = !(one->input_data[i] ^ one->weights[j * one->ni + i]);
            PRECISE_LOG("XOR %08X = %08X ^ %08X.", 
                    one->input_data[i] ^ one->weights[j * one->ni + i],
                    one->input_data[i], one->weights[j * one->ni + i]);
            PRECISE_LOG("XNOR %08X.", res);
            cast.gt = res;
            for (j = 0;j < BATCH;j++)
            {
                k += bit_cnt[cast.c[j]];
            }
        }
        PRECISE_LOG("%zu > %zu", k, one->beta);
        if (k > one->beta)
            SET_BIT(ret, j);
        PRECISE_LOG("Output %08X.", ret);
    }
    PRECISE_LOG("Output written %08X.", ret);
    *(one->output) = ret;
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
            nn_neuron_batch_activation_full(net, one);
            one->output = doubles_to_bits(one->output_full);
            if (!one->outputs)
            {
                net->outputs[j] = one->output;
            }
            DEBUG_PRINT("Layer %zd, neuron %zd: out(%d) \r\n\r", i, j, GET_BIT(one->output, j));
        }
    }
}

static void nn_neuron_batch_activation_full(const network *net, neuron_batch *one)
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
            for (k = 0;k < (INPUTS / BATCH);k++)
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

inline static void backward_batch_delta(neuron_batch *one, size_t j)
{
    size_t k,b,bk;
    for (b = 0;b < BATCH;b++)
    {
        DEBUG_PRINT("Batch neuron %zd start\r\n\r", b);
        one->delta[b] = 0.0;
        for (k = 0;k < (one->no / BATCH);k++)
        {
            DEBUG_PRINT("1\n");
            neuron_batch *gg = ((neuron_batch *)(one->outputs)) + k;
            DEBUG_PRINT("2\n");
            const float weight_f = gg->weights_full[j * BATCH + b];
            DEBUG_PRINT("3\n");
            for (bk = 0;bk < BATCH;bk++)
            {
                one->delta[b] += tanh(weight_f) * gg->delta[bk];
            }
            DEBUG_PRINT("4\n");
        }
        DEBUG_PRINT("ff %f\n", one->output_a_full[b]);
        one->delta[b] *= (1 - POW2(tanh(one->output_a_full[b])));
        DEBUG_PRINT("Batch neuron %zd: d(%f) \r\n\r", b, one->delta[b]);
    }
}

inline static void backward_batch_delta_last(neuron_batch *one, size_t j,
        group_type target[OUTPUTS / BATCH])
{
    size_t b;
    double bit;
    for (b = 0;b < BATCH;b++)
    {
        DEBUG_PRINT("Batch neuron %zd start last\r\n\r", b);
        bit = GET_BIT(target[j], b);
        one->delta[b] = (one->output_full[b] - bit);
        one->delta[b] *= (1 - POW2(tanh(one->output_a_full[b])));
        DEBUG_PRINT("Batch neuron %zd: d(%f) \r\n\r", b, one->delta[b]);
    }
}

inline static void backward_batch_weight(neuron_batch *one, double n)
{
    size_t k,b,bk;
    double sum;
    for (b = 0;b < BATCH;b++)
    {
        for (k = 0;k < one->ni;k++)
        {
            for (bk = 0;bk < BATCH;bk++)
            {
                DEBUG_PRINT("backward neuron %zd start\r\n\r", b);
                sum = n * one->delta[b] * PTR_CAST(one->inputs)[k].output_full[bk];
                one->weights_full[k * BATCH + bk] -= 
                        sum * (1 - POW2(tanh(one->weights_full[k * BATCH + bk])));
                DEBUG_PRINT("Batch neuron %zd, link %zd: w(%f) \r\n\r", 
                        b, k * BATCH + bk, one->weights_full[k * BATCH + bk]);
            }
        }
    }
}

inline static void backward_batch_weight_first(network *net, neuron_batch *one, double n)
{
    size_t k,b,bk;
    for (b = 0;b < BATCH;b++)
    {
        for (k = 0;k < one->ni;k++)
        {
            for (bk = 0;bk < BATCH;bk++)
            {
                DEBUG_PRINT("backward neuron first %zd start\r\n\r", b);
                one->weights_full[k * BATCH + bk] -= n * one->delta[b]
                        * GET_BIT(net->inputs[k], bk);
                DEBUG_PRINT("Batch neuron %zd, link %zd: w(%f) \r\n\r", 
                        b, k * BATCH + bk, one->weights_full[k * BATCH + bk]);
            }
        }
    }
}

inline static void backward_delta(network *net, group_type target[OUTPUTS / BATCH])
{
    size_t i,j;
    for (i = LAYERS;i > 0;i--)
    {
        DEBUG_PRINT("Layer %zd\r\n\r", i - 1);
        neuron_batch *line;
        line = net->hidden[i - 1];
        for (j = 0;j < net->hidden_cnt[i - 1];j++)
        {
            DEBUG_PRINT("Neuron batch %zd\r\n\r", j);
            neuron_batch *one;
            one = line + j;
            if (one->outputs)
                backward_batch_delta(one, j);
            else
                backward_batch_delta_last(one, j, target);
        }
    }
}

inline static void backward_weight(network *net)
{
    size_t i,j;
    for (i = 0;i < LAYERS;i++)
    {
        DEBUG_PRINT("Layer %zd\r\n\r", i - 1);
        neuron_batch *line;
        line = net->hidden[i];
        for (j = 0;j < net->hidden_cnt[i];j++)
        {
            DEBUG_PRINT("Neuron batch %zd\r\n\r", j);
            neuron_batch *one;
            one = line + j;
            if (one->inputs)
                backward_batch_weight(one, net->teaching_speed);
            else
                backward_batch_weight_first(net, one, net->teaching_speed);
        }
    }
}

void nn_backward(network *net, group_type target[OUTPUTS / BATCH])
{
    nn_inference_learning(net);
    // Delta propagation
    backward_delta(net, target);
    // Weights propagation
    backward_weight(net);
}
#endif

// Returns floating point random from -1.0 - 1.0.
static inline float frand()
{
    return (rand() / (float) (RAND_MAX / 2)) - 1.0;
}

static inline group_type gtrand()
{
    size_t j;
    group_type gtr = 0;
    for (j = 0;j < BATCH;j += 8)
    {
        gtr |= (rand() & 255U) << j;
    }
    PRECISE_LOG("%08X \n", gtr);
    return gtr;
}

group_type floats_to_bits(float *data)
{
    size_t i;
    group_type ret = 0;
    for (i = 0;i < BATCH;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

group_type doubles_to_bits(double *data)
{
    size_t i;
    group_type ret = 0;
    for (i = 0;i < BATCH;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}