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

#define MEMORY_STATIC
    
#define POW2(val) (val * val)
#define INPUTS 135
#define LAYERS 5
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
    #define LAYER_V_NEURONS 1
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_V_NEURONS
    #endif
#endif
#if (LAYERS > 3)
    #define LAYER_IV_NEURONS 4
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_IV_NEURONS
    #endif
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 16
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_III_NEURONS
    #endif
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 64
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_II_NEURONS
    #endif
#endif
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 128
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_I_NEURONS
    #endif
#endif

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
#ifndef MEMORY_STATIC
void mem_free(network *net);
#endif
float activation(const float a);
float pd_activation(const float a);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

