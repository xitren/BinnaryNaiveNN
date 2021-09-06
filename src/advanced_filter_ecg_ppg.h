/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   advanced_filter_ecg_ppg.h
 * Author: xitre
 *
 * Created on 5 сентября 2021 г., 3:24
 */

#ifndef ADVANCED_FILTER_ECG_PPG_H
#define ADVANCED_FILTER_ECG_PPG_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#define WINDOW 700
#define DOTS 4
//#define SIMD

typedef struct _tag_adv_filter adv_filter;
    
typedef void (*filter_parser)(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg);

struct _tag_adv_filter {
    // Window size.
    size_t window;
    // All the input data.
    uint16_t* ecg;
    uint16_t* ppg;
    // All the parsed data.
    uint16_t* ecg_parsed;
    uint16_t* ppg_parsed;
    // Max & min
    uint16_t min_ecg[3];
    uint16_t max_ecg[3];
    uint16_t min_ecg_it[3];
    uint16_t max_ecg_it[3];
    uint16_t min_ppg[3];
    uint16_t max_ppg[3];
    uint16_t min_ppg_it[3];
    uint16_t max_ppg_it[3];
    // Ring buffer
    size_t tail;
    size_t head;
    size_t head_parsed;
    // Parser func
    filter_parser fp;
    // Releaser.
    uint16_t* mark_ecg;
    uint16_t* mark_ppg;
    size_t mark_head;
};

void adv_filter_init(adv_filter* filt);
void adv_filter_input(adv_filter* filt, uint16_t data_ecg, uint16_t data_ppg);
void adv_filter_inputs(adv_filter* filt, uint16_t* data_ecg, uint16_t* data_ppg, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* ADVANCED_FILTER_ECG_PPG_H */

