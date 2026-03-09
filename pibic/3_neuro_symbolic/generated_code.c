#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    char buffer[10]; 
    strcpy(buffer, input); // BUG: Buffer overflow
}

int main() {
    char* input = calloc(20, 1);
    if(input) {
        parse_csv(input);
        free(input);
    }
    return 0;
}
