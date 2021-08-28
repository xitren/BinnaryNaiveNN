/* 
 * File:   genetic_search.c
 * Author: xitren
 *
 * Created on 27 августа 2021 г., 15:27
 */

#include "genetic_search.h"
#include "logger.h"
#include "binary_tools.h"
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

#define RAND_TRIGGER(prob) ((int)(RAND_MAX * prob))

static float err[POP_MAX];
static float pi[POP_MAX];
static float ki[POP_MAX];
static float copy[POP_MAX];
static size_t copy_fl[POP_MAX];
static size_t i_max;

static float sumf[5];
static float midf[5];
static float minf[5];

static group_type_gen genes_random[CROMOSOME_SIZE / BATCH];

static inline group_type_gen* genrand();
static void print_selection_table();

static void print_selection_table()
{
    size_t i;
    for (i = 0;i < i_max;i++)
    {
        TRACE_LOG("%1.3f %1.3f %1.3f %1.3f %02zu \n",
                err[i], pi[i], ki[i], copy[i], copy_fl[i]);
    }
    TRACE_LOG("%1.3f %1.3f %1.3f %1.3f %02zu \n",
            sumf[0], sumf[1], sumf[2], sumf[3], (size_t)sumf[4]);
    TRACE_LOG("%1.3f %1.3f %1.3f %1.3f %02zu \n",
            midf[0], midf[1], midf[2], midf[3], (size_t)midf[4]);
    TRACE_LOG("%1.3f %1.3f %1.3f %1.3f %02zu \n",
            minf[0], minf[1], minf[2], minf[3], (size_t)minf[4]);
}

void initiate_population_ranger(population_ranger* pop, error_f func, 
        size_t pop_initial, float mutation_prob, float crossover_prob)
{
    size_t i;
    // Tinn does not seed the random number generator.
    srand(time(0));
    for (i = 0;i < pop->pop_initial;i++)
    {
        memcpy(&(pop->pop_live[i].genes), genrand(),
                sizeof(group_type_gen) * CROMOSOME_SIZE / BATCH);
        pop->pop_live[i].population = 0;
    }
    pop->err = func;
    pop->pop_initial = pop_initial;
    pop->mutation_prob = mutation_prob;
    pop->crossover_prob = crossover_prob;
}

void pop_mutation_binary(_tag_chromosome_binary* mutant, float mutation_prob)
{
    size_t i,j;
    const size_t size = CROMOSOME_SIZE / BATCH;
    for (i = 0;i < size;i++)
    {
        for (j = 0;j < BATCH;j++)
        {
            if (rand() > RAND_TRIGGER(mutation_prob))
            {
                TGL_BIT(mutant->genes[i], j);
            }
        }
    }
}

void pop_crossover_multi_binary(_tag_chromosome_binary* child, 
        _tag_chromosome_binary* parentA, _tag_chromosome_binary* parentB)
{
    size_t i;
    const size_t size = CROMOSOME_SIZE / BATCH;
    const group_type_gen* switcher = genrand();
    for (i = 0;i < size;i++)
    {
        child->genes[i] = ((parentA->genes[i]) & (switcher[i])) |
                            ((parentB->genes[i]) & (!(switcher[i])));
    }
}

void pop_crossover_uno_binary(_tag_chromosome_binary* child, 
        _tag_chromosome_binary* parentA, _tag_chromosome_binary* parentB)
{
    size_t i;
    const size_t point = (size_t)((CROMOSOME_SIZE - 1) * (rand() / (float) (RAND_MAX)));
    const size_t k = point / BATCH;
    const size_t kz = point % BATCH;
    const size_t size = CROMOSOME_SIZE / BATCH;
    for (i = 0;i < k;i++)
    {
        child->genes[i] = parentA->genes[i];
    }
    child->genes[k] = 0;
    for (i = 0;(i < kz);i++)
    {
        const group_type_gen bit_tmp = (1U << i);
        child->genes[k] |= parentA->genes[k] & bit_tmp;
    }
    for (i = 0;(i < BATCH);i++)
    {
        const group_type_gen bit_tmp = (1U << i);
        child->genes[k] |= parentB->genes[k] & bit_tmp;
    }
    for (i = k + 1;i < size;i++)
    {
        child->genes[i] = parentB->genes[i];
    }
}

void pop_selection(population_ranger* pop)
{
    size_t i;
    i_max = pop->current;
    //Stage 1: Error calculation
    sumf[0] = 0;
    minf[0] = FLT_MAX;
    for (i = 0;i < i_max;i++)
    {
        sumf[0] += err[i] = pop->err(pop->pop_live[i]);
        if (sumf[0] < minf[0])
            minf[0] = sumf[0];
    }
    midf[0] = sumf[0] / i_max;
    //Stage 2: Normalized Pi
    sumf[1] = 0;
    minf[1] = FLT_MAX;
    for (i = 0;i < i_max;i++)
    {
        sumf[1] += pi[i] = err[i] / sumf[0];
        if (sumf[1] < minf[1])
            minf[1] = sumf[1];
    }
    midf[1] = sumf[1] / i_max;
    //Stage 3: Normalized Ki
    sumf[2] = 0;
    minf[2] = FLT_MAX;
    for (i = 0;i < i_max;i++)
    {
        sumf[2] += ki[i] = 2 * midf[1] - pi[i];
        if (sumf[2] < minf[2])
            minf[2] = sumf[2];
    }
    midf[2] = sumf[2] / i_max;
    //Stage 4: Next copy
    sumf[3] = 0;
    minf[3] = FLT_MAX;
    for (i = 0;i < i_max;i++)
    {
        sumf[3] += copy[i] = ki[i] / midf[2];
        if (sumf[3] < minf[3])
            minf[3] = sumf[3];
    }
    midf[3] = sumf[3] / i_max;
    //Stage 5: Floor
    sumf[4] = 0;
    minf[4] = FLT_MAX;
    for (i = 0;i < i_max;i++)
    {
        sumf[4] += copy_fl[i] = lround(copy[i]);
        if (sumf[4] < minf[4])
            minf[4] = sumf[4];
    }
    midf[4] = sumf[4] / i_max;
    print_selection_table();
}

// Returns random gene
static inline group_type_gen* genrand()
{
    size_t i,j;
    const size_t size = CROMOSOME_SIZE / BATCH;
    for (i = 0;i < size;i++)
    {
        genes_random[i] = 0;
        for (j = 0;j < BATCH;j += 8)
        {
            genes_random[i] |= (rand() & 255U) << j;
        }
    }
    return genes_random;
}