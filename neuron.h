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

#define INPUTS 5
#define LAYERS 4
#if (LAYERS > 0)
    #define LAYER_I_NEURONS 5
    #define OUTPUTS LAYER_I_NEURONS
#endif
#if (LAYERS > 1)
    #define LAYER_II_NEURONS 5
    #define OUTPUTS LAYER_II_NEURONS
#endif
#if (LAYERS > 2)
    #define LAYER_III_NEURONS 5
    #define OUTPUTS LAYER_III_NEURONS
#endif
#if (LAYERS > 3)
    #define LAYER_IV_NEURONS 5
    #define OUTPUTS LAYER_IV_NEURONS
#endif
#if (LAYERS > 4)
    #define LAYER_V_NEURONS 5
    #define OUTPUTS LAYER_V_NEURONS
#endif
#if (LAYERS > 5)
    #define LAYER_VI_NEURONS 5
    #define OUTPUTS LAYER_VI_NEURONS
#endif
#if (LAYERS > 6)
    #define LAYER_VII_NEURONS 5
    #define OUTPUTS LAYER_VII_NEURONS
#endif
#if (LAYERS > 7)
    #error "More than 7 layers unsupported!"
#endif

#define WEIGHTS 10

typedef float (*activation_f)(const neuron);

typedef struct _tag_neuron {
    // All the input weights.
    float* weights;
    // Input neurons.
    neuron* inputs;
    // Number of inputs.
    size_t ni;
    // Activation function.
    activation_f activator;
    // Propagation id.
    int propagator;
    // Output value
    float output;
    // Output neurons.
    neuron* outputs;
    // Number of outputs.
    size_t no;
} neuron;

typedef struct _tag_network {
    // All the input data.
    float inputs[INPUTS];
    // Hidden layers.
    neuron* hidden[LAYERS];
    // Hidden layers neuron count.
    const size_t hidden_cnt[] = {
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
    // All the output data.
    float outputs[OUTPUTS];
} network;

void nn_initialize(network net, activation_f activator);
void nn_inference(network net);

#ifdef __cplusplus
}
#endif

#endif /* NEURON_H */

