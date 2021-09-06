/* 
 * File:   data_reader.c
 * Author: xitre
 *
 * Created on 29 августа 2021 г., 16:16
 */

#include "data_reader.h"
#include "logger.h"
#include "binary_tools.h"
#include <string.h>
#include <stdio.h>

static size_t lns(FILE* const file, size_t* max_string);
static char* readln(FILE* const file, size_t size);
static float** new2d(const size_t rows, const size_t cols);
static void dfree(const data* d);
static void parse(data* data, char* line,
        const char* separator, const size_t row);

typedef union _tag_caster {
   uint32_t gt;
   uint8_t c[sizeof(uint32_t)];
} caster;  

// Returns the number of lines in a file.
static size_t lns(FILE* const file, size_t* max_string)
{
    int ch = EOF;
    size_t lines = 0;
    size_t str_sz = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        str_sz++;
        if(ch == '\n')
        {
            lines++;
            if (!(lines % 1000))
                PRECISE_LOG("Lines found %zu\n", lines);
            if ((*max_string) < str_sz)
                (*max_string) = str_sz;
            str_sz = 0;
        }
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    TRACE_LOG("Lines %zu, Max string size %zu\n", lines, str_sz);
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char* readln(FILE* const file, size_t size)
{
    int ch = EOF;
    size_t reads = 0;
    char* line = (char*) malloc((size) * sizeof(char));
    if (!line)
        ERROR_LOG("Read line memory allocation error!\n");
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
        {
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
            if (!line)
                ERROR_LOG("Line memory reallocation error!\n");
        }
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d(const size_t rows, const size_t cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    if (!row)
        ERROR_LOG("Rows memory allocation error!\n");
    for(size_t r = 0; r < rows; r++)
    {
        row[r] = (float*) malloc((cols) * sizeof(float));
        if (!(row[r]))
            ERROR_LOG("Distinct row memory allocation error!\n");
    }
    return row;
}

// Randomly shuffles a data object.
void shuffle(const data* d)
{
    for(int a = 0; a < d->rows; a++)
    {
        const size_t b = rand() % d->rows;
        float* ot = d->tg[a];
        float* it = d->in[a];
        // Swap output.
        d->tg[a] = d->tg[b];
        d->tg[b] = ot;
        // Swap input.
        d->in[a] = d->in[b];
        d->in[b] = it;
    }
    TRACE_LOG("Data freed\n");
}

// Frees a data object from the heap.
static void dfree(const data* d)
{
    for(size_t row = 0; row < d->rows; row++)
    {
        free(d->in[row]);
        free(d->tg[row]);
    }
    free(d->in);
    free(d->tg);
}

// New data object.
static void ndata(data* data, const size_t nips, const size_t nops, const size_t rows)
{
    data->in = new2d(rows, nips);
    data->tg = new2d(rows, nops);
    data->nips = nips;
    data->nops = nops;
    data->rows = rows;
}

// Gets one row of inputs and outputs from a string.
static void parse(data* data, char* line,
        const char* separator, const size_t row)
{
    const size_t cols = data->nips + data->nops;
    size_t col = 0;
    char *token;
    token = strtok(line, separator);
    while((token != NULL) && (col < cols))
    {
        const float val = atof(token);
        if (col < data->nips)
            data->in[row][col] = val;
        else
            data->tg[row][col - data->nips] = val;
        col++;
        token = strtok(NULL, separator);
    }
}

// Parses file from path getting all inputs and outputs for the neural network. 
// Returns data object.
void build_data(data* data, const char* path, const char* separator, 
        const size_t nips, const size_t nops)
{
    DESCRIBE_LOG("Opening data file\n");
    size_t max_line = 150;
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        ERROR_LOG("Could not open %s\n", path);
        return;
    }
    DESCRIBE_LOG("Countings rows\n");
//    const size_t rows = lns(file, &max_line);
    const size_t rows = 1000;
    DESCRIBE_LOG("Memory allocated\n");
    ndata(data, nips, nops, rows);
    DESCRIBE_LOG("Started data parsing\n");
    for(size_t row = 0; row < rows; row++)
    {
        char* line = readln(file, max_line);
        parse(data, line, separator, row);
        free(line);
        if (!(row % 1000))
            TRACE_LOG("Readed %d of %d\n", row, rows);
    }
    DESCRIBE_LOG("\nend\n");
    fclose(file);
    shuffle(data);
    return;
}

uint32_t floats_part_to_uint32(float *data, size_t n)
{
    size_t i;
    uint32_t ret = 0;
    for (i = 0;i < n;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint32_t floats_to_uint32(float *data)
{
    size_t i;
    uint32_t ret = 0;
    for (i = 0;i < 32U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint32_t doubles_part_to_uint32(double *data, size_t n)
{
    size_t i;
    uint32_t ret = 0;
    for (i = 0;i < n;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint32_t uint8_to_uint32(uint8_t *data)
{
    caster cast;
    size_t i;
    for (i = 0;i < 4U;i++)
    {
        cast.c[i] = data[i];
    }
    return cast.gt;
}

uint32_t floats_uint8_to_uint32(float *data)
{
    caster cast;
    size_t i;
    for (i = 0;i < 4U;i++)
    {
        cast.c[i] = (uint8_t)data[i];
    }
    return cast.gt;
}

uint32_t doubles_to_uint32(double *data)
{
    size_t i;
    uint32_t ret = 0;
    for (i = 0;i < 32U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint16_t floats_to_uint16(float *data)
{
    size_t i;
    uint16_t ret = 0;
    for (i = 0;i < 16U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint16_t doubles_to_uint16(double *data)
{
    size_t i;
    uint16_t ret = 0;
    for (i = 0;i < 16U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint8_t floats_to_uint8(float *data)
{
    size_t i;
    uint8_t ret = 0;
    for (i = 0;i < 8U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}

uint8_t doubles_to_uint8(double *data)
{
    size_t i;
    uint8_t ret = 0;
    for (i = 0;i < 8U;i++)
    {
        if (data[i] >= 0.5)
            SET_BIT(ret, i);
    }
    return ret;
}