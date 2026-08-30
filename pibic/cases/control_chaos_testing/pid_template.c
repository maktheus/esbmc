#include <stdlib.h>

extern float nondet_float();
extern void __ESBMC_assume(_Bool);
extern void __ESBMC_assert(_Bool, const char*);

void pid_step(float setpoint, float measured, float* integral, float* prev_error) {
    float kp = 0.5f; float ki = 0.1f; float kd = 0.05f;
    float error = setpoint - measured;
    *integral += error * 0.1f; 
    float derivative = (error - *prev_error) / 0.1f;
    *prev_error = error;
    
    float output = (kp * error) + (ki * *integral) + (kd * derivative);
    
    // Physical Actuator limit safety bound
    __ESBMC_assert(output >= -100.0f && output <= 100.0f, "PID Output exceeded physical bounds!");
}

int main() {
    float setpoint = 10.0f;
    float integral = 0.0f;
    float prev_error = 0.0f;
    
    // O perfil de caos injetado declara `noise` aqui. Nao ha fallback de
    // propósito: um `float noise = 0.0f;` como rede de seguranca foi o que
    // manteve os cinco perfis vacuos: a variavel injetada tinha outro nome e
    // ficava orfa, e o sistema era verificado sem ruido nenhum.
    // {{NOISE_INJECTION}}
    
    float measured_sensor = 0.0f + noise;
    
    for(int i = 0; i < 5; i++) {
        pid_step(setpoint, measured_sensor, &integral, &prev_error);
    }
    
    return 0;
}
