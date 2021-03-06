/* 
 * File:   genetic_search.h
 * Author: xitren
 *
 * Created on 27 августа 2021 г., 15:27
 */

#ifndef GENETIC_SEARCH_H
#define GENETIC_SEARCH_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <stdlib.h>
#include <stdint.h>
#include "bnn.h"

#define POP_MAX 200
//!!! Must be the power of 2
#define CHROMOSOME_SIZE ((HIDDEN_WEIGHTS_SIZE + BATCHES_SIZE) * BATCH)

#ifndef BATCH
#define BATCH 32
#endif
    
typedef uint32_t group_type_gen;

typedef struct _tag_chromosome_binary {
    group_type_gen genes[CHROMOSOME_SIZE / BATCH];
    size_t population;
} chromosome_binary;

typedef float (*error_f)(const chromosome_binary* const chr);

typedef struct _tag_population_ranger {
    size_t pop_initial;
    chromosome_binary pop_live[POP_MAX];
    error_f err;
    float err_calc[POP_MAX];
    float mutation_prob;
    float crossover_prob;
    float copy_roulette[POP_MAX];
} population_ranger;

void initiate_population_ranger(population_ranger* pop, error_f func, 
        size_t pop_initial, float mutation_prob, float crossover_prob);
void population_selection(population_ranger* pop);
void initiate_population(population_ranger* pop, chromosome_binary* init,
        error_f func, size_t pop_initial,
        float mutation_prob, float crossover_prob);

#ifdef __cplusplus
}
#endif

#endif /* GENETIC_SEARCH_H */

