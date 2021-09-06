/* 
 * File:   logger.h
 * Author: xitren
 *
 * Created on 27 августа 2021 г., 15:27
 */

#ifndef LOGGER_H
#define LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include <unistd.h>
#include <stdio.h>

#define STREAM 1
//#define FILE_PATH "run.log"
#define DEBUG_LVL 5

#ifdef STREAM
    #define WARNING "WARNING: "
    #define ERROR "ERROR: "
    #define TRACE "TRACE: "
    #define PRINT_PREPARE(lvl, tag, format, ... ) { log_size_glob = \
    snprintf( log_buf_glob, sizeof(log_buf_glob), "%s"format, tag, ## __VA_ARGS__  ); \
    logger_print_in_log(lvl);}
#if DEBUG_LVL >= 0
    #define DESCRIBE_LOG(format, ...) PRINT_PREPARE(0, "", format, ## __VA_ARGS__)
#else
    #define DESCRIBE_LOG(format, ...)
#endif
#if DEBUG_LVL >= 1
    #define ERROR_LOG(format, ...) PRINT_PREPARE(1, ERROR, format, ## __VA_ARGS__)
#else
    #define ERROR_LOG(format, ...)
#endif
#if DEBUG_LVL >= 2
    #define WARNING_LOG(format, ...) PRINT_PREPARE(2, WARNING, format, ## __VA_ARGS__)
#else
    #define WARNING_LOG(format, ...)
#endif
#if DEBUG_LVL >= 3
    #define DEBUG_LOG(format, ...) PRINT_PREPARE(3, "", format, ## __VA_ARGS__)
#else
    #define DEBUG_LOG(format, ...)
#endif
#if DEBUG_LVL >= 4
    #define TRACE_LOG(format, ...) PRINT_PREPARE(4, TRACE, format, ## __VA_ARGS__)
#else
    #define TRACE_LOG(format, ...)
#endif
#if DEBUG_LVL >= 5
    #define PRECISE_LOG(format, ...) PRINT_PREPARE(5, "", format, ## __VA_ARGS__)
#else
    #define PRECISE_LOG(format, ...)
#endif
#endif

typedef _READ_WRITE_RETURN_TYPE (*print_function)(int __fd, const void *__buf, size_t __nbyte);

extern char log_buf_glob[128];
extern size_t log_size_glob;

void logger_print_in_log(int lvl);

#ifdef STREAM
inline void logger_change_stream_printer(print_function new_one);
#endif

#ifdef FILE_PATH
void logger_open_file();
void logger_close_file();
inline void logger_change_file(char* filename);
inline void logger_change_file_printer(print_function new_one);
#endif

#ifdef __cplusplus
}
#endif

#endif /* LOGGER_H */

