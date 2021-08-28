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

#define POP_MAX 1024
//!!! Must be the power of 2
#define CROMOSOME_SIZE 1024
#define BATCH 32
    
typedef uint32_t group_type_gen;

typedef struct _tag_chromosome_binary {
    group_type_gen genes[CROMOSOME_SIZE / BATCH];
    size_t population;
} chromosome_binary;

typedef struct _tag_population_ranger {
    size_t pop_initial;
    chromosome_binary pop_live[POP_MAX];
    error_f err;
    size_t current;
    float mutation_prob;
    float crossover_prob;
} population_ranger;

typedef float (*error_f)(const chromosome_binary* const chr);

void initiate_chromosome_binary(chromosome_binary* chr);

void initiate_population_ranger(population_ranger* pop, error_f func);
void pop_selection(population_ranger* pop);
void pop_crossover(population_ranger* pop);
void pop_mutation(population_ranger* pop);

void pop_crossover_uno_binary(_tag_chromosome_binary* child, 
        _tag_chromosome_binary* parentA, _tag_chromosome_binary* parentB);
void pop_crossover_multi_binary(_tag_chromosome_binary* child, 
        _tag_chromosome_binary* parentA, _tag_chromosome_binary* parentB);
void pop_mutation_binary(_tag_chromosome_binary* mutant, float mutation_prob);

#ifdef __cplusplus
}
#endif

#endif /* GENETIC_SEARCH_H */

