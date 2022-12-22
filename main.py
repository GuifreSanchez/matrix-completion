

import classes  as cl
import generate as gen
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
from math import sqrt
from time import time

main_colors = ["r","b","c", "g", "m", "k"]

def print_results(n_results_global):
    n = len(n_results_global)
    print("mean_time,mean_iters")
    for i in range(n):
        print(mean_results(n_results_global[i]))

def mean_results(results_global):
    results = np.array(results_global)
    mean_time = np.mean(results[:, gen.INDEX_RESULTS_TIME_ELAPSED])
    mean_iterations = np.mean(results[:, gen.INDEX_RESULTS_ITERATIONS])
    return mean_time, mean_iterations

def homotopyPlot(n = 8000, k_OS = 20, OS = 8, k_max = 20): 
    # Generate OISs
    Omega = gen.genRandomOmega(n,k_OS,OS)
    Lambda = gen.genRandomOmega(n,k_OS,OS)
    print("Omega, Lambda OIS generated")

    # Generate matrix with discretization of f
    A_hom = gen.square_matrix_from_function(n,gen.f)
    print("f discretization complete")

    # Compute results without homotopy
    results_not_hom = gen.LRGeomCGGlobal(n, k_max, Omega, Lambda, A_hom, homotopy = False)

    # Compute results with homotopy 
    results = gen.LRGeomCGGlobal(n, k_max, Omega, Lambda, A_hom, homotopy = True)

    # Print results 
    print("results_not_hom array")
    print(results_not_hom)
    print("results array")
    print(results)

    # Set plot data
    ranks = np.array([k for k in range(1,k_max + 1)])
    rel_error = np.array(results[:,gen.INDEX_RESULTS_REL_LAMBDA])
    rel_error_not_hom = np.array(results_not_hom[:, gen.INDEX_RESULTS_REL_LAMBDA])

    # Generate plot
    fig, ax = plt.subplots()
    lineNotHom, = ax.plot(ranks, rel_error_not_hom, linestyle="dashed", marker="o", markersize = 4, label = 'LRGeomCG (no hom)', color='black')
    lineHom, = ax.plot(ranks, rel_error, linestyle="solid", marker="o", markersize = 4, label = 'LRGeomCG (hom)')
    ax.legend()
    ax.set_yscale("log")
    ax.set(xlabel='rank', ylabel='relative error', title='')
    ax.grid()
    file_name = "homotopy_kOS_" + str(k_OS) + "_n_" + str(n) + "_k_max_" + str(k_max) + "_OS_" + str(OS) + ".png"
    fig.savefig(file_name)
    plt.show()


def OSPlot(n = 1000, k = 30, samples = 3, OS0 = 3, OS1 = 5, nOS = 11, max_iterations = 100):
    print("Fixed n, k: ", n, k)
    OSs = np.linspace(OS0, OS1, nOS)
    # Results for all samples
    full_results = []
    # Results (mean) for each tuple (n, k, OS)
    global_results = []
    for os in OSs:
        print("\n")
        print("n, k, OS = ", n, k, os)
        results = gen.LRGeomCGNSamples(n,k,OS=os, samples=samples, max_iterations = max_iterations, prints = False)
        full_results.append(results[1])
        global_results.append(results[0])

    # Print global results
    print_results(global_results)

    # Generate plot
    fig, ax = plt.subplots()
    color0 = [0.0, 0.0, 0.5]
    color1 = [0.0, 1.0, 1.0]
    new_color = [0.0, 0.0, 0.0]
    for i in range(len(OSs)):
        results_i = full_results[i]
        # print(results_i.shape)
        step = float(i) / len(OSs)
        for c in range(3):
            new_color[c] = color0[c] + (color1[c] - color0[c]) * step
        # print(new_color)
        for j in range(len(results_i)):
            relative_residuals = results_i[j]
            iters = np.array([i for i in range(len(relative_residuals))])
            line, = ax.plot(iters, relative_residuals, color=(new_color[0], new_color[1], new_color[2]))
            if (j == 0): line.set_label("OS = " + str(OSs[i]))
        ax.legend()

    plt.yscale("log")
    title_string = 'Convergence curves n = ' + str(n) + ', k = ' + str(k)
    ax.set(xlabel='iterations', ylabel='relative residual', title=title_string)
    ax.grid()
    file_name = "os_"
    for os in OSs:
        file_name = file_name + str(os) + "_"
    file_name = file_name + "fixed_n_" + str(n) + "_fixed_k_" + str(k) + ".png"
    fig.savefig(file_name)
    plt.show()

def rankPlot(n = 1000, ks = [30, 40, 50], samples = 3, OS = 4, max_iterations = 200):
    print("Fixed n, OS: ", n, OS)
    # Results for all samples
    full_results = []
    # Results (mean) for each tuple (n, k, OS)
    global_results = []
    for k in ks:
        print("\n")
        print("n, k, OS = ", n, k, OS)
        results = gen.LRGeomCGNSamples(n,k,OS=OS, samples=samples, max_iterations = max_iterations, prints = False)
        full_results.append(results[1])
        global_results.append(results[0])

    # Print global results
    print_results(global_results)

    # Generate plot
    fig, ax = plt.subplots()
    for i in range(len(ks)):
        results_i = full_results[i]
        # print(results_i.shape)
        for j in range(len(results_i)):
            relative_residuals = results_i[j]
            iters = np.array([i for i in range(len(relative_residuals))])
            line, = ax.plot(iters, relative_residuals, color=main_colors[i])
            if (j == 0): line.set_label("k = " + str(ks[i]))
        ax.legend()

    plt.yscale("log")
    title_string = 'Convergence curves n = ' + str(n) + ', OS = ' + str(OS)
    ax.set(xlabel='iterations', ylabel='relative residual', title=title_string)
    ax.grid()
    file_name = "k_"
    for k in ks:
        file_name = file_name + str(k) + "_"
    file_name = file_name + "fixed_n_" + str(n) + "_fixed_os_" + str(OS) + ".png"
    fig.savefig(file_name)
    plt.show()

def sizePlot(ns = [500, 1000, 1500, 2000], k = 20, samples = 3, OS = 4, max_iterations = 200):
    print("Fixed k, OS: ", k, OS)
    # Results for all samples
    full_results = []
    # Results (mean) for each tuple (n, k, OS)
    global_results = []
    for n in ns:
        print("\n")
        print("n, k, OS = ", n, k, OS)
        results = gen.LRGeomCGNSamples(n,k,OS=OS, samples=samples, max_iterations = max_iterations, prints = False)
        full_results.append(results[1])
        global_results.append(results[0])

    # Print global results
    print_results(global_results)

    # Generate plot
    fig, ax = plt.subplots()
    for i in range(len(ns)):
        results_i = full_results[i]
        # print(results_i.shape)
        for j in range(len(results_i)):
            relative_residuals = results_i[j]
            iters = np.array([i for i in range(len(relative_residuals))])
            line, = ax.plot(iters, relative_residuals, color=main_colors[i])
            if (j == 0): line.set_label("n = " + str(ns[i]))
        ax.legend()

    plt.yscale("log")
    title_string = 'Convergence curves k = ' + str(k) + ', OS = ' + str(OS)
    ax.set(xlabel='iterations', ylabel='relative residual', title=title_string)
    ax.grid()
    file_name = "n_"
    for n in ns:
        file_name = file_name + str(n) + "_"
    file_name = file_name + "fixed_k_" + str(k) + "_fixed_os_" + str(OS) + ".png"
    fig.savefig(file_name)
    plt.show()


# rank 10-15 plot
rankPlot(n = 8000, ks = [15, 14, 13, 12, 11, 10], samples = 10, OS = 3, max_iterations = 300)

# OSs plot
OSPlot(n = 8000, k = 10, samples = 5, OS0 = 3, OS1 = 5, nOS = 11, max_iterations=300)

# Rank 20-60 plot
rankPlot(n = 8000, ks = [60, 50, 40, 30, 20], OS = 3, samples = 10, max_iterations = 200)

# Homotopy vs. non-homotopy plot
homotopyPlot(n = 8000, k_OS = 20, OS = 8, k_max = 20)

# Size 1000-16000 plot
sizePlot(ns =[1000, 2000, 4000, 8000, 16000], k=40, samples = 10, OS=3, max_iterations=200)



