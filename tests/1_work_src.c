#include "neuron.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
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
    const int cols = data.nips;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, "\t"));
        data.in[row][col] = val;
    }
}

// Gets one row of inputs and outputs from a string.
static void parse_targ(const Data data, char* line, const int row)
{
    const int cols = data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, "\t"));
        data.tg[row][col] = val;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build(const char* path, const char* path_targ, const int nips, const int nops)
{
    PRINT("Opening data file\n");
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        PRINT("Could not open %s\n", path);
        exit(1);
    }
    PRINT("Opening target file\n");
    FILE* file_targ = fopen(path_targ, "r");
    if(file_targ == NULL)
    {
        PRINT("Could not open %s\n", path_targ);
        exit(1);
    }
    PRINT("Countings rows\n");
    const int rows = 170650;
    PRINT("Memory allocated\n");
    Data data = ndata(nips, nops, rows);
    PRINT("Started data parsing\n");
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
        line = readln(file_targ);
        parse_targ(data, line, row);
        free(line);
        if (!(row % 1000))
            PRINT("Readed %d of %d\n", row, rows);
    }
    PRINT("\nend\n");
    fclose(file);
    fclose(file_targ);
    return data;
}

// Computes error.
static float err(const float a, const float b)
{
    return 0.5f * (a - b) * (a - b);
}

// Computes total error of target to output.
static float toterr(const float* const tg, const float* const o, const int size)
{
    float sum = 0.0f;
    for(int i = 0; i < size; i++)
        sum += err(tg[i], o[i]);
    return sum;
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

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 1400;
    const int nops = 3;
    float target[3];
    
    // Load the training set.
    PRINT("Read started\n");
    const Data data = build("tests/input_datap.txt", "tests/target_datap.txt", nips, nops);
    PRINT("Files readed\n");
    
    // Train, baby, train.
    network net;
    PRINT("Initialization started\n");
    nn_initialize(&net,&activation,&pd_activation);
    net.teaching_speed = 4;
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
            for (int i = 0; i < (nips/2); i++)
            {
                net.inputs[i] = data.in[row][i*2];
            }
            for (int i = 0; i < OUTPUTS; i++)
            {
                target[i] = data.tg[row][i];
            }
            DEBUG_PRINT("Target: %f %f %f\n", target[0], target[1], target[2]);
            nn_backward(&net,target);
            DEBUG_PRINT("Outputs: %f %f %f\n", net.outputs[0], net.outputs[1], net.outputs[2]);
            error += toterr(target, net.outputs, OUTPUTS);
        }
        net.teaching_speed *= 0.994f;
        PRINT("%d) error %.12f :: learning rate %f\n",
            it,
            (double) error / data.rows,
            (double) net.teaching_speed);
        nn_save(&net, "net_usial_pressure2.txt");
    }
    dfree(data);
    return 0;
}
