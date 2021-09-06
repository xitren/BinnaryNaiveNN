/* 
 * File:   data_reader.h
 * Author: xitre
 *
 * Created on 29 августа 2021 г., 16:16
 */

#ifndef DATA_READER_H
#define DATA_READER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

// Data object.
typedef struct _tag_data
{
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    size_t nips;
    // Number of outputs to neural network.
    size_t nops;
    // Number of rows in file (number of sets for neural network).
    size_t rows;
} data;

void build_data(data* data, const char* path, const char* separator, 
        const size_t nips, const size_t nops);
void shuffle(const data* d);
uint32_t floats_part_to_uint32(float *data, size_t n);
uint32_t doubles_part_to_uint32(double *data, size_t n);
uint32_t floats_to_uint32(float *data);
uint32_t doubles_to_uint32(double *data);
uint16_t floats_to_uint16(float *data);
uint16_t doubles_to_uint16(double *data);
uint8_t floats_to_uint8(float *data);
uint8_t doubles_to_uint8(double *data);
uint32_t uint8_to_uint32(uint8_t *data);
uint32_t floats_uint8_to_uint32(float *data);

#ifdef __cplusplus
}
#endif

#endif /* DATA_READER_H */

