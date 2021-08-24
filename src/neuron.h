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

//#define DEBUG_PRINT( format, ... ) { size_glob = \
//snprintf( buf_glob, sizeof(buf_glob), format, ## __VA_ARGS__  ); \
//write(1, buf_glob, size_glob); }
#define DEBUG_PRINT( format, ... ) 
#define PRINT( format, ... ) { log_size_glob = \
snprintf( log_buf_glob, sizeof(log_buf_glob), format, ## __VA_ARGS__  ); \
write(1, log_buf_glob, log_size_glob); }
#define POW2(val) (val * val)
#define INPUTS 1400
#define LAYERS 3
#if (LAYERS > 7)
    #error "More than 7 layers unsupported!"
#endif
#if (LAYERS > 6)
    #define LAYER_VII_NEURONS 5
    #define OUTPUTS LAYER_VII_NEURONS
#endif
#if (LAYERS > 5)
    #define LAYER_VI_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_VI_NEURONS
    #endif
#endif
#if (LAYERS > 4)
    #define LAYER_V_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_V_NEURONS
    #endif
#endif
#if (LAYERS > 3)
    #define LAYER_IV_NEURONS 3
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_IV_NEURONS
    #endif
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 3
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_III_NEURONS
    #endif
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 16
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_II_NEURONS
    #endif
#endif
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 64
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_I_NEURONS
    #endif
#endif

extern char log_buf_glob[128];
extern size_t log_size_glob;    

typedef float (*activation_f)(const float a);
typedef float (*pd_activation_f)(const float a);

typedef struct _tag_neuron {
    // All the input weights.
    float* weights;
    // Input neurons.
    void* inputs;
    // Number of inputs.
    size_t ni;
    // Bias value
    float bias;
    // Delta value
    float delta;
    // Activation function.
    activation_f activator;
    // Activation function.
    pd_activation_f pd_activator;
    // Propagation id.
    int propagator;
    // Output value
    float output;
    // Output neurons.
    void* outputs;
    // Number of outputs.
    size_t no;
} neuron;

typedef struct _tag_network {
    // All the input data.
    float inputs[INPUTS];
    // Hidden layers.
    neuron* hidden[LAYERS];
    // Hidden layers neuron count.
    size_t* hidden_cnt;
    // All the output data.
    float outputs[OUTPUTS];
    // Teaching speed
    float teaching_speed;
} network;

void nn_initialize(network *net, activation_f activator, pd_activation_f pd_activator);
void nn_inference(network *net);
void nn_backward(network *net, float target[OUTPUTS]);
void nn_save(network *net, const char* path);
void nn_load(network *net, const char* path);
float activation(const float a);
float pd_activation(const float a);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

