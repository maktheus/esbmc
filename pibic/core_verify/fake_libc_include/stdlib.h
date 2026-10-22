#ifndef _STDLIB_H_
#define _STDLIB_H_
typedef unsigned long long size_t;
void* malloc(size_t size);
void free(void* ptr);
void* calloc(size_t nmemb, size_t size);
void* realloc(void* ptr, size_t size);
int abs(int j);
double atof(const char *nptr);
int atoi(const char *nptr);
int rand(void);
void srand(unsigned int seed);
void abort(void);
void exit(int status);
#endif
