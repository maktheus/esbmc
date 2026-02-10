#include <stdio.h>
#include <assert.h>

// ESBMC intrinsics for non-determinism
int nondet_int();
float nondet_float();

// System Parameters
#define TARGET_TEMP 100.0f
#define MAX_SAFE_TEMP 150.0f
#define HEATING_RATE 0.1f  // Degrees per power unit
#define COOLING_RATE 2.0f  // Degrees per tick
#define DT 1.0f            // Time step

// Control Parameters (PID)
#define Kp 1.0f
#define Ki 0.1f
#define Kd 0.5f

float plant_update(float current_temp, float heater_power) {
    // Basic thermal model: T_new = T_old + (HeatIn - HeatOut) * dt
    float heating = heater_power * HEATING_RATE;
    float cooling = COOLING_RATE; // Simplified constant cooling
    
    // Physical constraint: Temp cannot drop below ambient (say 20.0)
    // For simplicity, we just apply heating/cooling
    float new_temp = current_temp + (heating - cooling) * DT;
    if (new_temp < 20.0f) new_temp = 20.0f;
    
    return new_temp;
}

float get_sensor_reading(float true_temp) {
    // CHAOS INJECTION: Simulate sensor noise/fault
    // The sensor reading might be slightly off due to interference
    
    // Nondeterministic noise between -NOISE_MAX and +NOISE_MAX
    // We'll let ESBMC choose the noise value to find worst-case scenarios
    float noise = nondet_float();
    __ESBMC_assume(noise >= -5.0f && noise <= 5.0f); // 5 degree noise range
    
    return true_temp + noise;
}

int main() {
    float temp = 25.0f; // Initial temp
    float integral = 0.0f;
    float prev_error = 0.0f;
    
    // Simulation loop
    // Unrolling 10 steps for bounded verification
    for (int i = 0; i < 10; ++i) {
        // 1. Sense
        float measured_temp = get_sensor_reading(temp);
        
        // 2. Compute Control (PID)
        float error = TARGET_TEMP - measured_temp;
        integral += error * DT;
        float derivative = (error - prev_error) / DT;
        
        float output = Kp * error + Ki * integral + Kd * derivative;
        
        // Actuator saturation (Heater 0-100%)
        if (output > 100.0f) output = 100.0f;
        if (output < 0.0f) output = 0.0f;
        
        // Safety Interlock: Cutoff heater if temperature is critical (despite PID)
        if (measured_temp > 120.0f) {
            output = 0.0f;
        }
        
        prev_error = error;
        
        // 3. Actuate (Update Plant)
        temp = plant_update(temp, output);
        
        // 4. Verify Safety Property
        // The system must NEVER exceed safe temperature despite sensor noise
        assert(temp < MAX_SAFE_TEMP);
    }
    
    return 0;
}
