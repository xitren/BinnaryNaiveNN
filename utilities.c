#include "utilities.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Computes error.
inline float error(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Returns partial derivative of error function.
inline float derivative(const float a, const float b)
{
    return a - b;
}

// Computes total error of target to output.
inline float error_vector(const float* const a, const float* const b, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += error(a[i], b[i]);
    return sum;
}

// Activation function.
float activation(const float a)
{
    return 1.0f / (1.0f + expf(-a));
}

// Returns partial derivative of activation function.
inline float activation_derivative(const float a)
{
    return a * (1.0f - a);
}

// Returns floating point random from 0.0 - 1.0.
inline float frand()
{
    return rand() / (float) RAND_MAX;
}