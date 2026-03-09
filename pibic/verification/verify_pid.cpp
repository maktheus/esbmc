#include <assert.h>
#include "Arduino.h"
#include "famous_pid/PID_v1.h"

unsigned long current_millis = 0;
unsigned long millis() {
    return current_millis;
}

double nondet_double();
void __ESBMC_assume(int condition);

int main() {
    double Setpoint = 100.0;
    double Input = 25.0;
    double Output = 0.0;
    
    double DT = 1.0;
    double HEATING_RATE = 0.1;
    double COOLING_RATE = 2.0;

    PID myPID(&Input, &Output, &Setpoint, 1.0, 0.1, 0.5, DIRECT);
    myPID.SetMode(AUTOMATIC);
    myPID.SetOutputLimits(0, 100);
    myPID.SetSampleTime(1000); // 1000 ms = 1 sec
    
    double real_temp = 25.0;
    
    for (int k = 0; k < 10; k++) {
        current_millis += 1000;
        
        double noise = nondet_double();
        __ESBMC_assume(noise >= -5.0 && noise <= 5.0);
        
        Input = real_temp + noise;
        
        if (Input > 120.0) {
            Output = 0.0;
        } else {
            myPID.Compute();
        }
        
        double heating = Output * HEATING_RATE;
        real_temp = real_temp + (heating - COOLING_RATE) * DT;
        if (real_temp < 20.0) real_temp = 20.0;
        
        assert(real_temp < 150.0);
    }
    
    return 0;
}
