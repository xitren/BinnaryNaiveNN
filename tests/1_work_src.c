#include "neuron.h"
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
    int size = 128;
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
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        data.in[row][col] = val;
    }
}

// Gets one row of inputs and outputs from a string.
static void parse_targ(const Data data, char* line, const int row)
{
    const int cols = data.nops;
    for(int col = 0; col < cols; col++)
    {
        const float val = atof(strtok(col == 0 ? line : NULL, " "));
        data.tg[row][col] = val;
    }
}

// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build(const char* path, const char* path_targ, const int nips, const int nops)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        exit(1);
    }
    FILE* file_targ = fopen(path_targ, "r");
    if(file_targ == NULL)
    {
        printf("Could not open %s\n", path_targ);
        exit(1);
    }
    const int rows = lns(file);
    const int rows_targ = lns(file_targ);
    if(rows != rows_targ)
    {
        printf("Files has different lines numbers\n");
        exit(1);
    }
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
        line = readln(file_targ);
        parse_targ(data, line, row);
        free(line);
    }
    fclose(file);
    fclose(file_targ);
    return data;
}

// Learns and predicts hand written digits with 98% accuracy.
int main(void)
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    
    const int nips = 700;
    const int nops = 3;
    float target[3];
    
    // Load the training set.
    const Data data = build("tests/input_datap.txt", "tests/target_datap.txt", nips, nops);
    printf("Files readed\n");
    
    // Train, baby, train.
    network net;
    nn_initialize(&net,&activation,&pd_activation);
    nn_save(&net, "net2.txt");
    for (int it = 0; it < 10; it++){
        for (int row = 0; row < data.rows; row++){
            for (int i = 0; i < nips; i++){
                net.inputs[i] = data.in[row][i];
            }
            for (int i = 0; i < 3; i++){
                target[i] = data.tg[row][i];
            }
            nn_backward(&net,target);
        }
    }
    nn_save(&net, "net.txt");
    return 0;
}
