
#include <stdlib.h>
#include <string.h>

void parse_csv(char* input) {
    // FIX: Dynamic allocation or bounds check
    // Here we just use strncpy for safety
    char buffer[10];
    strncpy(buffer, input, 9);
    buffer[9] = '\0';
}

int main() {
    char* input = malloc(20);
    parse_csv(input);
    free(input);
    return 0;
}
    