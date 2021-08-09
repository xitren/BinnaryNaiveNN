/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.c
 * Author: xitre
 *
 * Created on 20 июня 2021 г., 17:21
 */

#include <stdio.h>
#include <stdlib.h>
#include "neuron.h"

/*
 * 
 */
int main(int argc, char** argv) {
    network net;
    float target[OUTPUTS];
    target[0] = 0.0f;
    target[1] = 0.0f;
    nn_initialize(&net, &activation, &pd_activation);
    nn_backward(&net, target);
    nn_backward(&net, target);
    return (EXIT_SUCCESS);
}

