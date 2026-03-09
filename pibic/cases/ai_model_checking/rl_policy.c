#include <stdlib.h>

extern float nondet_float();
extern void __ESBMC_assume(_Bool);
extern void __ESBMC_assert(_Bool, const char*);

// A simple trained Actor-Critic policy network output bound
// Let's assume the continuous action space MUST be within [-1.0, 1.0] (e.g., steering a vehicle)

void check_rl_policy_safety() {
    // Simulated State Space (e.g., LIDAR distance, velocity)
    float pos_x = nondet_float();
    float pos_y = nondet_float();
    float velocity = nondet_float();
    
    // Valid physical constraints for the environment
    __ESBMC_assume(pos_x >= 0.0f && pos_x <= 100.0f);
    __ESBMC_assume(pos_y >= 0.0f && pos_y <= 100.0f);
    __ESBMC_assume(velocity >= -10.0f && velocity <= 10.0f);
    
    // Abstracted Neural Network computations for Actor (Steering)
    // Intentionally injecting a bug where high velocity + edge pos_y causes output > 1.0
    // This simulates a poorly generalized RL policy at edge cases.
    float steering_action = (pos_y * 0.01f) + (velocity * 0.05f); 
    
    // Action MUST be strictly between -1.0 and 1.0 to ensure physical safety
    __ESBMC_assert(steering_action >= -1.0f && steering_action <= 1.0f, "RL Policy Action Exceeded Safe Actuator Limits!");
}

int main() {
    check_rl_policy_safety();
    return 0;
}
