# matrix-completion

Upon execution, main.py should generate the following plots, in the following order:

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

Notice that some of this plots may take a long time (see running times in the report) to be generated. 