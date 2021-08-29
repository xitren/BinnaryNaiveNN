/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   bitcnt_table.c
 * Author: xitre
 *
 * Created on 29 августа 2021 г., 3:34
 */

#include <stdio.h>
#include <stdlib.h>
#include "binary_tools.h"
#include "logger.h"

/*
 * 
 */
int main(int argc, char** argv) {
    size_t i,j,k;
    DESCRIBE_LOG("{");
    for (i = 0;i < 256;i++)
    {
        k = 0;
        for (j = 0;j < 8;j++)
        {
            if (GET_BIT(i, j))
                k++;
        }
        DESCRIBE_LOG("%zu, ", k);
    }
    DESCRIBE_LOG("}");
    return (EXIT_SUCCESS);
}

