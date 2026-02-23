#include <stdio.h>
#include "../verification/nn_forward_pass.c"

int main() {
    float input[INPUT_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f};
    run_network(input);
    printf("Input: {0.0, 0.0, 0.0, 0.0}\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("Output[%d]: %f\n", i, output[i]);
    }
    return 0;
}
