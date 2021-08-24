#include "bnn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Data object.
typedef struct
{
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    int nips;
    // Number of outputs to neural network.
    int nops;
    // Number of rows in file (number of sets for neural network).
    int rows;
}
Data;

// Returns the number of lines in a file.
static int lns(FILE* const file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char* readln(FILE* const file)
{
    int ch = EOF;
    int reads = 0;
    int size = 12800;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d(const int rows, const int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata(const int nips, const int nops, const int rows)
{
    const Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse(const Data data, char* line, const int row)
{
    const int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

// Frees a data object from the heap.
static void dfree(const Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}

// Randomly shuffles a data object.
static void shuffle(const Data d)
{
    for(int a = 0; a < d.rows; a++)
    {
        const int b = rand() % d.rows;
        float* ot = d.tg[a];
        float* it = d.in[a];
        // Swap output.
        d.tg[a] = d.tg[b];
        d.tg[b] = ot;
        // Swap input.
        d.in[a] = d.in[b];
        d.in[b] = it;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build(const char* path, const int nips, const int nops)
{
    PRINT("Opening data file\n");
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        PRINT("Could not open %s\n", path);
        PRINT("Get it from the machine learning database: ");
        PRINT("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        exit(1);
    }
    PRINT("Countings rows\n");
    const int rows = lns(file);
    PRINT("Memory allocated\n");
    Data data = ndata(nips, nops, rows);
    PRINT("Started data parsing\n");
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
        if (!(row % 1000))
            PRINT("Readed %d of %d\n", row, rows);
    }
    PRINT("\nend\n");
    fclose(file);
    return data;
}

// Computes error.
static float err(const group_type a, const group_type b)
{
    float sum = 0.0f;
    for(size_t k = 0; k < sizeof(group_type); k++)
    {
        if (GET_BIT(a, k) != GET_BIT(b, k))
        {
            sum += 1.;
        }
    }
    return sum;
}

// Computes total error of target to output.
static float toterr(const group_type* const tg, const group_type* const o, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
    {
        sum += err(tg[i], o[i]);
    }
    return sum;
}

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 256;
    const int nops = 32;
    group_type target[OUTPUTS / BATCH];
    
    // Load the training set.
    PRINT("Read started\n");
    const Data data = build("tests/semeion.data", nips, 10);
    PRINT("Files readed\n");
    
    // Train, baby, train.
    network net;
    PRINT("Initialization started\n");
    nn_initialize(&net);
    net.teaching_speed = 10;
    float error = 0.9 * data.rows;
    PRINT("Learning started\n");
    for (int it = 0; (it < 10000)
            && (net.teaching_speed > 0.001) 
            && ((error / data.rows) > 0.01); it++)
    {
        shuffle(data);
        error = 0.;
        for (int row = 0; row < data.rows; row++)
        {
            for (int i = 0;i < (INPUTS / BATCH);i++)
            {
                net.inputs[i] = floats_to_bits(data.in[row] + i * BATCH);
            }
            target[0] = 0;
            for (int i = 0;i < 10;i++)
            {
                if (data.tg[row][i] > 0.5)
                    SET_BIT(target[0], i);
            }
            DEBUG_PRINT("Target: %08X\n", target[0]);
            nn_backward(&net, target);
            DEBUG_PRINT("Outputs: %08X\n", net.outputs[0]);
            error += toterr(target, net.outputs, OUTPUTS / BATCH);
        }
        net.teaching_speed *= 0.99f;
        PRINT("%d) error %.12f :: learning rate %f\n",
            it,
            (double) error / data.rows,
            (double) net.teaching_speed);
    }
    dfree(data);
    return 0;
}
