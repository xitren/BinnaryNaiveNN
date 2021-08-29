/* 
 * File:   logger.c
 * Author: xitren
 *
 * Created on 27 августа 2021 г., 15:27
 */

#include "logger.h"
#include <stdlib.h>
#include <fcntl.h>

#define WARNING_BUFFER_OVERFLOW WARNING"Printout buffer overflow!\n"
#define WARNING_FILE_NOT_CLOSED WARNING"Cant close log file!\n"
#define ERROR_FILE_NOT_FOUND ERROR"Cant open log file!\n"

char log_buf_glob[128];
size_t log_size_glob;
char* file_path = 0;
int file = 0;
print_function stream_printer = &write;
print_function file_printer = &write;

inline void logger_change_stream_printer(print_function new_one)
{
    stream_printer = new_one;
}

#ifdef FILE_PATH
inline void logger_change_file(char* filename)
{
    file_path = filename;
    logger_close_file();
    logger_open_file();
}

inline void logger_change_file_printer(print_function new_one)
{
    file_printer = new_one;
}

void logger_open_file()
{
    if (!file_path)
    {
        file = open(FILE_PATH, O_WRONLY| O_CREAT);
    }
    else
    {
        file = open(file_path, O_WRONLY| O_CREAT);
    }
    if (file < 0) {
#ifdef STREAM
#if DEBUG_LVL >= 1
        write(STREAM, ERROR_FILE_NOT_FOUND, sizeof(ERROR_FILE_NOT_FOUND));
#endif
#endif
        return;
    }
}

void logger_close_file()
{
    if (file > 0)
    {
        if (close(file) < 0)
        {
#ifdef STREAM
#if DEBUG_LVL >= 2
            write(STREAM, WARNING_FILE_NOT_CLOSED, sizeof(WARNING_FILE_NOT_CLOSED));
#endif
#endif
        }
    }
}
#endif

void logger_print_in_log(int lvl)
{
#ifdef STREAM
#if DEBUG_LVL >= 2
    if (log_size_glob >= sizeof(log_buf_glob))
    {
        write(STREAM, WARNING_BUFFER_OVERFLOW, sizeof(WARNING_BUFFER_OVERFLOW));
    }
#endif
    write(STREAM, log_buf_glob, log_size_glob);
#endif
#ifdef FILE_PATH
    if (file == 0)
    {
        logger_open_file();
    }
    if (file > 0)
    {
#if DEBUG_LVL >= 2
        if (log_size_glob >= sizeof(log_buf_glob))
        {
            write(file, WARNING_BUFFER_OVERFLOW, sizeof(WARNING_BUFFER_OVERFLOW));
        }
#endif
        write(file, log_buf_glob, log_size_glob);
    }
#endif
}
