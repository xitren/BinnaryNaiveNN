/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   utilities.h
 * Author: xitre
 *
 * Created on 16 июня 2021 г., 2:05
 */

#ifndef UTILITIES_H
#define UTILITIES_H

#ifdef __cplusplus
extern "C" {
#endif

// Computes error.
inline float error(const float a, const float b);
// Returns partial derivative of error function.
inline float derivative(const float a, const float b);
// Computes total error of target to output.
inline float error_vector(const float* const a, const float* const b, const int size);
// Activation function.
float activation(const float a);
// Returns partial derivative of activation function.
inline float activation_derivative(const float a);
// Returns floating point random from 0.0 - 1.0.
inline float frand();

#ifdef __cplusplus
}
#endif

#endif /* UTILITIES_H */

