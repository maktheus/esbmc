import matplotlib.pyplot as plt
import csv
import os

def plot_case2():
    sizes = []
    times = []
    
    csv_path = "pibic/results/case2_benchmark.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            sizes.append(int(row[0]))
            times.append(float(row[1]))
            
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker='o', linestyle='-', color='b')
    plt.title('Case 2: Verification Time vs Matrix Dimension')
    plt.xlabel('Matrix Dimension (NxN)')
    plt.ylabel('Verification Time (s)')
    plt.grid(True)
    plt.savefig('pibic/results/case2_plot.png')
    print("Generated pibic/results/case2_plot.png")

def plot_case3():
    iterations = []
    times = []
    
    csv_path = "pibic/results/case3_agent_stats.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            iterations.append(int(row[0]))
            times.append(float(row[2]))
            
    plt.figure(figsize=(8, 5))
    plt.bar(iterations, times, color='green')
    plt.title('Case 3: Verification Time per Agent Iteration')
    plt.xlabel('Agent Iteration')
    plt.ylabel('Verification Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('pibic/results/case3_plot.png')
    print("Generated pibic/results/case3_plot.png")

if __name__ == "__main__":
    plot_case2()
    plot_case3()
