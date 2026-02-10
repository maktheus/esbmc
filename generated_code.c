
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    // BUG: Fixed size buffer, input can be larger
    char buffer[10]; 
    strcpy(buffer, input);
}

int main() {
    char* input = malloc(20);
    // Abstract input
    parse_csv(input);
    free(input);
    return 0;
}
    