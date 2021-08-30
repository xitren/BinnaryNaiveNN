/* 
 * File:   genetic_search.c
 * Author: xitren
 *
 * Created on 27 августа 2021 г., 15:27
 */

#include "genetic_search.h"
#include "logger.h"
#include "binary_tools.h"
#include <float.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define RAND_TRIGGER(prob) ((int)(RAND_MAX * prob))

static group_type_gen genes_random[CHROMOSOME_SIZE / BATCH];

static inline group_type_gen* genrand();
static inline void print_selection_table(float err[POP_MAX], float pi[POP_MAX],
        float ki[POP_MAX], float copy[POP_MAX], float copy_r[POP_MAX],
        float sumf[4], float midf_pi, float midf_ki);
static void calculate_roulette(population_ranger* pop);
static void pop_crossover_uno_binary(chromosome_binary* child, 
        chromosome_binary* parentA, chromosome_binary* parentB);
static void pop_crossover_multi_binary(chromosome_binary* child, 
        chromosome_binary* parentA, chromosome_binary* parentB);
static void pop_mutation_binary(chromosome_binary* mutant, float mutation_prob);
static chromosome_binary* select_from_roulette(population_ranger* pop);
static void swap_sort(population_ranger* pop);
static inline void print_chromosome(chromosome_binary* chr);
static inline float frand();
static inline void print_err_table(population_ranger* pop);

void initiate_population_ranger(population_ranger* pop, error_f func, 
        size_t pop_initial, float mutation_prob, float crossover_prob)
{
    size_t i;
    // Tinn does not seed the random number generator.
    srand(time(0));
    for (i = 0;i < POP_MAX;i++)
    {
        memcpy(&(pop->pop_live[i].genes), genrand(),
                sizeof(group_type_gen) * (CHROMOSOME_SIZE / BATCH));
        PRECISE_LOG("Initial populated %zd\n", i);
        print_chromosome(pop->pop_live + i);
        pop->pop_live[i].population = 0;
    }
    pop->err = func;
    pop->pop_initial = pop_initial;
    pop->mutation_prob = mutation_prob;
    pop->crossover_prob = crossover_prob;
}

void pop_selection(population_ranger* pop)
{
    size_t i;
    chromosome_binary pop_live;
    memset(&pop_live, 0, sizeof(chromosome_binary));
    swap_sort(pop);
    calculate_roulette(pop);
    print_err_table(pop);
    for (i = 1;i < POP_MAX;i++)
        pop->copy_roulette[i] += pop->copy_roulette[i - 1];
    PRECISE_LOG("Selection 1.\n");
    const chromosome_binary* parentA = select_from_roulette(pop);
    PRECISE_LOG("Selection 2.\n");
    chromosome_binary* parentB = select_from_roulette(pop);
    while (parentB == parentA)
        parentB = select_from_roulette(pop);
    if (!parentA || !parentB)
        return;
    pop_crossover_multi_binary(&pop_live, parentA, parentB);
    for (i = 0;i < POP_MAX;i++)
    {
        if (frand() < pop->mutation_prob)
        {
            pop_mutation_binary(pop->pop_live + i, pop->crossover_prob);
            PRECISE_LOG("%zu Mutated! \n", i);
            pop->err_calc[i] = pop->err((const chromosome_binary* const)(pop->pop_live + i));
            break;
        }
    }
    pop_live.population = parentA->population + 1;
    print_chromosome(&pop_live);
    pop->pop_live[POP_MAX - 1] = pop_live;
    pop->err_calc[POP_MAX - 1] = pop->err((const chromosome_binary* const)
            (pop->pop_live + POP_MAX - 1));
    print_err_table(pop);
    PRECISE_LOG("=================================================\n");
}

static void swap_sort(population_ranger* pop)
{
    size_t temp;
    size_t i,j;
    static size_t swapper[POP_MAX];
    static float err_calc[POP_MAX];
    static float copy_roulette[POP_MAX];
    static chromosome_binary pop_live[POP_MAX];
    for (i = 0;i < POP_MAX;i++)
    {
        swapper[i] = i;
    }
    for (i = 0;i < POP_MAX;i++)
    {
        for (j = 1;j < POP_MAX;j++)
        {
            if (pop->err_calc[swapper[j - 1]] > pop->err_calc[swapper[j]])
            {
                temp = swapper[j - 1];
                swapper[j - 1] = swapper[j];
                swapper[j] = temp;
            }
        }
    }
    memcpy(pop_live, pop->pop_live, sizeof(pop_live));
    memcpy(err_calc, pop->err_calc, sizeof(err_calc));
    memcpy(copy_roulette, pop->copy_roulette, sizeof(copy_roulette));
    for (i = 0;i < POP_MAX;i++)
    {
        pop->pop_live[i] = pop_live[swapper[i]];
        pop->err_calc[i] = err_calc[swapper[i]];
        pop->copy_roulette[i] = copy_roulette[swapper[i]];
    }
}

// Returns floating point random 1.0.
static inline float frand()
{
    return (rand() / (float) (RAND_MAX));
}

static chromosome_binary* select_from_roulette(population_ranger* pop)
{
    size_t j;
    const float rn = frand();
    for (j = 0;j < POP_MAX;j++)
    {
        PRECISE_LOG("rn %f < %f(%zu)\n", rn, pop->copy_roulette[j], j);
        if (rn < (pop->copy_roulette[j]))
            break;
    }
    if (j == POP_MAX)
    {
        ERROR_LOG("Roulette critical error!\n");
        return 0;
    }
    PRECISE_LOG("Selected %zu from roulette.\n", j);
    return (pop->pop_live + j);
}

static void pop_mutation_binary(chromosome_binary* mutant, float mutation_prob)
{
    size_t i,j;
    const size_t size = CHROMOSOME_SIZE / BATCH;
    PRECISE_LOG("Before mutating\n");
//    print_chromosome(mutant);
    for (i = 0;i < size;i++)
    {
        for (j = 0;j < BATCH;j++)
        {
            if (frand() < mutation_prob)
            {
                TGL_BIT(mutant->genes[i], j);
            }
        }
    }
    PRECISE_LOG("After mutating\n");
//    print_chromosome(mutant);
}

static void pop_crossover_multi_binary(chromosome_binary* child, 
        chromosome_binary* parentA, chromosome_binary* parentB)
{
    size_t i;
    const size_t size = CHROMOSOME_SIZE / BATCH;
    const group_type_gen* switcher = genrand();
    PRECISE_LOG("Crossover\n");
    PRECISE_LOG("Parent A\n");
//    print_chromosome(parentA);
    PRECISE_LOG("Parent B\n");
//    print_chromosome(parentB);
    for (i = 0;i < size;i++)
    {
        child->genes[i] = ((parentA->genes[i]) & (switcher[i])) |
                            ((parentB->genes[i]) & (~(switcher[i])));
    }
    PRECISE_LOG("Child\n");
//    print_chromosome(child);
}

static void pop_crossover_uno_binary(chromosome_binary* child, 
        chromosome_binary* parentA, chromosome_binary* parentB)
{
    size_t i;
    const size_t point = (size_t)((CHROMOSOME_SIZE - 1) * (rand() / (float) (RAND_MAX)));
    const size_t k = point / BATCH;
    const size_t kz = point % BATCH;
    const size_t size = CHROMOSOME_SIZE / BATCH;
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

static void calculate_roulette(population_ranger* pop)
{
    size_t i;
    static float err[POP_MAX];
    static float pi[POP_MAX];
    static float ki[POP_MAX];
    static float copy[POP_MAX];
    float sumf[4];
    float midf_pi, midf_ki;
    //Stage 1: Error calculation
    sumf[0] = 0;
    for (i = 0;i < POP_MAX;i++)
    {
        err[i] = pop->err((const chromosome_binary* const)(pop->pop_live + i));
        pop->err_calc[i] = err[i];
        sumf[0] += err[i];
    }
    //Stage 2: Normalized Pi
    sumf[1] = 0;
    for (i = 0;i < POP_MAX;i++)
        sumf[1] += pi[i] = err[i] / sumf[0];
    midf_pi = sumf[1] / POP_MAX;
    //Stage 3: Normalized Ki
    sumf[2] = 0;
    for (i = 0;i < POP_MAX;i++)
        sumf[2] += ki[i] = 2 * midf_pi - pi[i];
    midf_ki = sumf[2] / POP_MAX;
    //Stage 4: Next copy
    sumf[3] = 0;
    for (i = 0;i < POP_MAX;i++)
        sumf[3] += copy[i] = ki[i] / midf_ki;
    //Stage 5: Floor
    for (i = 0;i < POP_MAX;i++)
        pop->copy_roulette[i] = copy[i] / sumf[3];
    print_selection_table(err, pi, ki, copy, pop->copy_roulette, sumf, midf_pi, midf_ki);
}

// Returns random gene
static inline group_type_gen* genrand()
{
    size_t i,j;
    const size_t size = CHROMOSOME_SIZE / BATCH;
    PRECISE_LOG("Random generated gene\n");
    for (i = 0;i < size;i++)
    {
        genes_random[i] = 0;
        for (j = 0;j < BATCH;j += 8)
        {
            genes_random[i] |= ((group_type_gen)(frand() * 255U)) << j;
        }
        PRECISE_LOG("%08X-", genes_random[i]);
    }
    PRECISE_LOG("\n");
    return genes_random;
}

static inline void print_chromosome(chromosome_binary* chr)
{
    size_t i;
    PRECISE_LOG("Chromosome ");
    for (i = 0;i < (CHROMOSOME_SIZE / BATCH);i++)
    {
        PRECISE_LOG("%08X-", chr->genes[i]);
    }
    PRECISE_LOG("\n");
}

static inline void print_selection_table(float err[POP_MAX], float pi[POP_MAX],
        float ki[POP_MAX], float copy[POP_MAX], float copy_r[POP_MAX],
        float sumf[4], float midf_pi, float midf_ki)
{
    size_t i;
    PRECISE_LOG("Selection table\n");
    for (i = 0;i < POP_MAX;i++)
    {
        PRECISE_LOG("%02zu) %1.3f %1.3f %1.3f %1.3f %1.3f \n",
                i, err[i], pi[i], ki[i], copy[i], copy_r[i]);
    }
    PRECISE_LOG("%1.3f %1.3f %1.3f %1.3f \n", sumf[0], sumf[1], sumf[2], sumf[3]);
    PRECISE_LOG("%1.3f %1.3f %1.3f %1.3f \n", 0., midf_pi, midf_ki, 0.);
    PRECISE_LOG("========================\n");
}

static inline void print_err_table(population_ranger* pop)
{
    size_t i;
    float mid = 0., min = pop->err_calc[0];
    PRECISE_LOG("Error table\n");
    for (i = 0;i < POP_MAX;i++)
    {
        PRECISE_LOG("%02zu) %1.3f \n", i, pop->err_calc[i]);
        mid += pop->err_calc[i];
        if (min > pop->err_calc[i])
            min = pop->err_calc[i];
    }
    mid /= POP_MAX;
    TRACE_LOG("Minimal err: %1.2f. Middle err: %1.2f.\n", min, mid);
    PRECISE_LOG("========================\n");
}