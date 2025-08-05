import nashpy as nash
import numpy as np
import random
import time


def generateMatrix(dim):
    min = 0
    max = 100

    list = [[None] * dim] * dim

    for i in range(dim):
        for j in range(dim):
            list[i][j] = random.randrange(min, max + 1)

    return np.array(list)


DIM = 5000

A = generateMatrix(DIM)
B = generateMatrix(DIM)

start = time.process_time_ns()

rps = nash.Game(A, B)
eqs = rps.support_enumeration()

stop = time.process_time_ns()

print("Execution time of nashpy equilibrium with", DIM, "size:",
      int((stop - start) / 1e6), "ms")
