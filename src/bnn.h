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
#include <stdlib.h>
#include <math.h>
#include "binary_tools.h"

#define LEARNER
typedef uint32_t group_type;
#define BATCH 32
#define DEBUG_PRINT( format, ... ) { log_size_glob = \
snprintf( log_buf_glob, sizeof(log_buf_glob), format, ## __VA_ARGS__  ); \
write(1, log_buf_glob, log_size_glob); }
#define PRINT( format, ... ) { log_size_glob = \
snprintf( log_buf_glob, sizeof(log_buf_glob), format, ## __VA_ARGS__  ); \
write(1, log_buf_glob, log_size_glob); }
    
#define INPUTS 256
#if ((INPUTS % BATCH) > 0)
    #error "Not a full butch!"
#endif
#define LAYERS 2
#if (LAYERS > 7)
    #error "More than 7 layers unsupported!"
#endif
#if (LAYERS > 6)
    #define LAYER_VII_NEURONS 32
    #if ((LAYER_VII_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #define OUTPUTS LAYER_VII_NEURONS
#endif
#if (LAYERS > 5)
    #define LAYER_VI_NEURONS 32
    #if ((LAYER_VI_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_VI_NEURONS
    #endif
#endif
#if (LAYERS > 4)
    #define LAYER_V_NEURONS 32
    #if ((LAYER_V_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_V_NEURONS
    #endif
#endif
#if (LAYERS > 3)
    #define LAYER_IV_NEURONS 32
    #if ((LAYER_IV_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_IV_NEURONS
    #endif
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 32
    #if ((LAYER_III_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_III_NEURONS
    #endif
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 32
    #if ((LAYER_II_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_II_NEURONS
    #endif
#endif
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 64
    #if ((LAYER_I_NEURONS % BATCH) > 0)
        #error "Not a full butch!"
    #endif
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_I_NEURONS
    #endif
#endif

typedef struct _tag_neuron_batch {
    // All the input weights.
    group_type* weights;
#ifdef LEARNER
    double* weights_full;
#endif
    // Input neurons.
    void* inputs;
    // Number of inputs.
    size_t ni;
    // Bias value
    group_type bias;
#ifdef LEARNER
    double bias_full[BATCH];
    // Delta value
    double delta[BATCH];
#endif
    // Propagation id.
    int propagator;
    // Output value
    group_type output;
#ifdef LEARNER
    double output_a_full[BATCH];
    double output_full[BATCH];
#endif
    // Output neurons.
    void* outputs;
    // Number of outputs.
    size_t no;
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
#ifdef LEARNER
void nn_backward(network *net, group_type target[OUTPUTS / BATCH]);
#endif
group_type floats_to_bits(float *data);
group_type doubles_to_bits(double *data);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

