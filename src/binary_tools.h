/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   binary_tools.h
 * Author: xitre
 *
 * Created on 29 августа 2021 г., 0:33
 */

#ifndef BINARY_TOOLS_H
#define BINARY_TOOLS_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <stdint.h>

#define GET_BIT(num, n) ((num >> n) & 1U)
#define SET_BIT(num, n) (num |= (1U << n))
#define CLR_BIT(num, n) (num &= ~(1U << n))
#define TGL_BIT(num, n) (num ^= (1U << n))
#define COPY_BIT(type, to, from, n) {const type bit_tmp = (1U << n); \
                                to &= ~(bit_tmp); \
                                to |= from & bit_tmp;}


#ifdef __cplusplus
}
#endif

#endif /* BINARY_TOOLS_H */

