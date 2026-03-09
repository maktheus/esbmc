class ChaosGenerator:
    """
    Mathematical Noise Injector mapping pure mathematical distributions
    into ESBMC syntax limits.
    """
    
    @staticmethod
    def get_uniform_noise(bound=1.0):
        # Maps nondet_float() to strictly uniform [-bound, bound]
        return f"float noise_uniform = nondet_float();\n__ESBMC_assume(noise_uniform >= -{bound}f && noise_uniform <= {bound}f);\n"

    @staticmethod
    def get_gaussian_noise(mean=0.0, std=1.0):
        # A simple bounded triangular approximation mapped to SMT constraints 
        # because exact Box-Muller transformations hit non-linear solver limits
        bound = std * 3.0 # 99% interval
        return f"float noise_gauss = nondet_float();\n__ESBMC_assume(noise_gauss >= {mean-bound}f && noise_gauss <= {mean+bound}f);\n"
        
    @staticmethod
    def get_impulse_noise(magnitude=100.0, probability=0.01):
        # Simulates binary extreme bounds failure for sensors
        return f"float noise_impulse = nondet_float();\n__ESBMC_assume(noise_impulse == 0.0f || noise_impulse == {magnitude}f || noise_impulse == -{magnitude}f);\n"

    @staticmethod
    def get_drift_noise(rate=0.1, max_time=100):
        # Linear degradation assuming time index T
        return f"float noise_drift = nondet_float();\n__ESBMC_assume(noise_drift >= 0.0f && noise_drift <= ({rate}f * {max_time}f));\n"
        
    @staticmethod
    def get_sinusoidal_noise(amplitude=1.0):
        # Pure peak-to-peak bound check, since Sin wave solving in bitwuzla/Z3 is too expensive
        return f"float noise_sin = nondet_float();\n__ESBMC_assume(noise_sin >= -{amplitude}f && noise_sin <= {amplitude}f);\n"

def inject_pid_chaos(template_path, output_path, noise_type="Uniform"):
    mapper = {
        "Uniform": ChaosGenerator.get_uniform_noise(),
        "Gaussian": ChaosGenerator.get_gaussian_noise(),
        "Impulse": ChaosGenerator.get_impulse_noise(),
        "Drift": ChaosGenerator.get_drift_noise(),
        "Sinusoidal": ChaosGenerator.get_sinusoidal_noise()
    }
    
    with open(template_path, 'r') as f:
        content = f.read()
        
    # Replace the {{NOISE_INJECTION}} symbol in the C stub with formal mathematical properties
    content = content.replace("// {{NOISE_INJECTION}}", mapper[noise_type])
    
    with open(output_path, 'w') as f:
        f.write(content)
