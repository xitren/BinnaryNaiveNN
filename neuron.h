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
#include "utilities.h"

#define INPUTS 5
#define LAYERS 4
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
    #define LAYER_IV_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_IV_NEURONS
    #endif
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_III_NEURONS
    #endif
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_II_NEURONS
    #endif
#endif
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 5
    #ifndef OUTPUTS
        #define OUTPUTS LAYER_I_NEURONS
    #endif
#endif

#define WEIGHTS 10

typedef float (*activation_f)(const network, const neuron);

typedef struct _tag_neuron {
    // All the input weights.
    float* weights;
    // Input neurons.
    struct neuron* inputs;
    // Number of inputs.
    size_t ni;
    // Bias value
    float bias;
    // Activation function.
    activation_f activator;
    // Propagation id.
    int propagator;
    // Output value
    float output;
    // Output neurons.
    struct neuron* outputs;
    // Number of outputs.
    size_t no;
} neuron;

typedef struct _tag_network {
    // All the input data.
    float inputs[INPUTS];
    // Hidden layers.
    struct neuron* hidden[LAYERS];
    // Hidden layers neuron count.
    size_t* hidden_cnt;
    // All the output data.
    float outputs[OUTPUTS];
} network;

void nn_initialize(network net, activation_f activator);
void nn_inference(network net);
float nn_sigma_activation(const network net, const neuron one);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

