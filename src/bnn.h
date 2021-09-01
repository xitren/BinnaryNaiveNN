/* 
 * File:   neuron.h
 * Author: xitre
 *
 * Created on 16 июня 2021 г., 2:10
 */

#ifndef NEURON_H
#define NEURON_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "binary_tools.h"

//#define LEARNER
typedef uint32_t group_type;
#define BATCH 32
    
#define INPUTS 11200
#if ((INPUTS % BATCH) > 0)
    #error "Not a full butch!"
#endif
#define LAYERS 4
#if (LAYERS > 7)
    #error "More than 7 layers unsupported!"
#endif
#if (LAYERS > 6)
    #define LAYER_VII_NEURONS 32
    #if ((LAYER_VII_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #define OUTPUTS LAYER_VII_NEURONS
#else
    #define LAYER_VII_NEURONS 0
#endif
#if (LAYERS > 5)
    #define LAYER_VI_NEURONS 32
    #if ((LAYER_VI_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_VI_NEURONS
    #endif
#else
    #define LAYER_VI_NEURONS 0
#endif
#if (LAYERS > 4)
    #define LAYER_V_NEURONS 11200
    #if ((LAYER_V_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_V_NEURONS
    #endif
#else
    #define LAYER_V_NEURONS 0
#endif
#if (LAYERS > 3)
    #define LAYER_IV_NEURONS 1024
    #if ((LAYER_IV_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_IV_NEURONS
    #endif
#else
    #define LAYER_IV_NEURONS 0
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 512
    #if ((LAYER_III_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_III_NEURONS
    #endif
#else
    #define LAYER_III_NEURONS 0
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 1024
    #if ((LAYER_II_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_II_NEURONS
    #endif
#else
    #define LAYER_II_NEURONS 0
#endif
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 4096
    #if ((LAYER_I_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_I_NEURONS
    #endif
#else
    #define LAYER_I_NEURONS 0
#endif

#define HIDDEN1_WEIGHTS ((LAYER_I_NEURONS / BATCH) * INPUTS)
#define HIDDEN2_WEIGHTS ((LAYER_II_NEURONS / BATCH) * LAYER_I_NEURONS)
#define HIDDEN3_WEIGHTS ((LAYER_III_NEURONS / BATCH) * LAYER_II_NEURONS)
#define HIDDEN4_WEIGHTS ((LAYER_IV_NEURONS / BATCH) * LAYER_III_NEURONS)
#define HIDDEN5_WEIGHTS ((LAYER_V_NEURONS / BATCH) * LAYER_IV_NEURONS)
#define HIDDEN6_WEIGHTS ((LAYER_VI_NEURONS / BATCH) * LAYER_V_NEURONS)
#define HIDDEN7_WEIGHTS ((LAYER_VII_NEURONS / BATCH) * LAYER_VI_NEURONS)

#define HIDDEN_WEIGHTS_SIZE (HIDDEN1_WEIGHTS + HIDDEN2_WEIGHTS \
            + HIDDEN3_WEIGHTS + HIDDEN4_WEIGHTS + HIDDEN5_WEIGHTS \
            + HIDDEN6_WEIGHTS + HIDDEN7_WEIGHTS)
#define BATCHES_SIZE ((LAYER_I_NEURONS + LAYER_II_NEURONS \
            + LAYER_III_NEURONS + LAYER_IV_NEURONS + LAYER_V_NEURONS \
            + LAYER_VI_NEURONS + LAYER_VII_NEURONS) / BATCH)

typedef struct _tag_neuron_batch {
    // All the input weights.
    group_type* weights;
    // Input data.
    group_type* input_data;
    // Number of inputs.
    size_t ni;
    // Beta value
    size_t beta;
    // Output value
    group_type* output;
    // Number of outputs.
    size_t no;
#ifdef LEARNER
    // Bias value
    group_type bias;
    double* weights_full;
    // Input neurons.
    void* inputs;
    double bias_full[BATCH];
    // Delta value
    double delta[BATCH];
    // Propagation id.
    int propagator;
    double output_a_full[BATCH];
    double output_full[BATCH];
    // Output neurons.
    void* outputs;
#endif
} neuron_batch;

typedef struct _tag_network {
    // All the input data.
    group_type inputs[INPUTS / BATCH];
    // Hidden layers.
    neuron_batch* hidden[LAYERS];
    // Hidden layers neuron count.
    size_t* hidden_cnt;
    // All the output data.
    group_type outputs[OUTPUTS / BATCH];
    // Teaching speed
    double teaching_speed;
} network;

extern char log_buf_glob[128];
extern size_t log_size_glob;    

void nn_initialize(network *net);
void nn_inference(network *net);
float nn_error(network *net, group_type *inputs[INPUTS / BATCH], 
        group_type *outputs[OUTPUTS / BATCH], size_t n);
void nn_set_beta(network *net, group_type *betas);
void nn_set_weights(group_type* weights);
void nn_save(network *net, const char* path);
void nn_load(network *net, const char* path);
#ifdef LEARNER
void nn_backward(network *net, group_type target[OUTPUTS / BATCH]);
#endif

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

