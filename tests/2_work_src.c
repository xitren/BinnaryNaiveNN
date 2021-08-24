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
        if (!(row % 100))
            PRINT(".");
    }
    PRINT("\nend\n");
    fclose(file);
    fclose(file_targ);
    return data;
}

static void result_save(Data *data, const char* path)
{
    size_t i,j;
    FILE* const file = fopen(path, "w");
    for (i = 0;i < data->rows;i++)
    {
        for (j = 0;j < data->nips;j++)
        {
            fprintf(file, "%f\t", one->bias);
        }
        fprintf(file, "\n");
    }
    fclose(file);
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
    PRINT("Read started\n");
    const Data data = build("tests/input_datap.txt", "tests/target_datap.txt", nips, nops);
    PRINT("Files readed\n");
    
    // Train, baby, train.
    network net;
    nn_initialize(&net,&activation,&pd_activation);
    nn_load(&net, "net.txt");
    for (int row = 0; row < data.rows; row++){
        for (int i = 0; i < nips; i++){
            net.inputs[i] = data.in[row][i];
        }
        nn_inference(&net);
        for (int i = 0; i < nops; i++){
            data.tg[row][i] = net.outputs[i];
        }
    }
    result_save(&data, "results.txt");
    return 0;
}