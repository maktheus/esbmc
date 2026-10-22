#ifndef _STDIO_H_
#define _STDIO_H_
typedef unsigned long long size_t;
typedef void FILE;
extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;
int printf(const char *format, ...);
int sprintf(char *str, const char *format, ...);
int snprintf(char *str, size_t size, const char *format, ...);
int puts(const char *s);
#endif
