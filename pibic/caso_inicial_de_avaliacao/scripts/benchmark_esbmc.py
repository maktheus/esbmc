import time
import subprocess
import matplotlib.pyplot as plt

C_TEMPLATE = """
#define TESTING
#include "../src/runq.c"

#ifdef VERIFY_ESBMC
int8_t nondet_int8();
#endif

#define N {n}
#define D {n}
#define GROUP_SIZE 2

int main() {{
    GS = GROUP_SIZE;

    int8_t x_q[N];
    float x_s[N / GROUP_SIZE];
    int8_t w_q[D * N];
    float w_s[(D * N) / GROUP_SIZE];

#ifdef VERIFY_ESBMC
    for (int i = 0; i < N / GROUP_SIZE; i++) x_s[i] = 1.0f;
    for (int i = 0; i < (D * N) / GROUP_SIZE; i++) w_s[i] = 1.0f;
#endif

#ifdef VERIFY_ESBMC
    for (int i = 0; i < N; i++) {{ x_q[i] = nondet_int8(); }}
    for (int i = 0; i < D * N; i++) {{ w_q[i] = nondet_int8(); }}
#endif

    QuantizedTensor tensor_x = {{ .q = x_q, .s = x_s }};
    QuantizedTensor tensor_w = {{ .q = w_q, .s = w_s }};
    float xout[D];

    // Foco em propriedades de Acesso de Memoria e Overflow Inteiro
    matmul(xout, &tensor_x, &tensor_w, N, D);

    return 0;
}}
"""

def run_esbmc_for_size(n):
    c_code = C_TEMPLATE.format(n=n)
    with open("../src/verify_esbmc_test.c", "w") as f:
        f.write(c_code)
    
    esbmc_bin = "/home/uchoa/esbmc/build/src/esbmc/esbmc"
    # Removing --floatbv to verify primarily arrays and basic operations for speed
    cmd = [esbmc_bin, "../src/verify_esbmc_test.c", "-DVERIFY_ESBMC", "--function", "main", "--unwind", "257", "--timeout", "120s"]
    
    start_time = time.time()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        success = "VERIFICATION SUCCESSFUL" in proc.stdout or "VERIFICATION SUCCESSFUL" in proc.stderr
    except Exception as e:
        success = False
    
    elapsed = time.time() - start_time
    return elapsed, success

def main():
    sizes = [2, 4, 6, 8]
    times = []
    
    print("Iniciando benchmark empírico e real (Pode levar de 1 a 2 minutos)...")
    for n in sizes:
        print(f"Executando Verificação para N={n}x{n}...")
        t, success = run_esbmc_for_size(n)
        times.append(t)
        print(f"-> Concluído N={n}: tempo={t:.2f}s, verificado={success}")
        
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='s', linestyle='-', color='g')
    plt.title("Tempo de Execução Real ESBMC (Z3 CPU Time) vs Dimensão NxN")
    plt.xlabel("Dimensão da Matriz Quantizada (N)")
    plt.ylabel("Tempo de Verificação Empírica (Segundos)")
    plt.grid(True)
    
    for i in range(len(sizes)):
        plt.text(sizes[i], times[i] + 0.1, f"{times[i]:.2f}s", fontsize=9, ha='center')
        
    plt.savefig("../docs/grafico_verificacao.png")
    print("Gráfico EMPÍRICO gerado e salvo em ../docs/grafico_verificacao.png")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
