# pipe the output to big.txt to generate a 2.4 GB big ASCII file
import numpy as np

n = 100000000

print("# POINTS:",n)
print("depth gravity")

x = np.linspace(0, 6371000,n)
y = np.random.random_sample((n,))+9.8

it = np.nditer(x, flags=['f_index'])
while not it.finished:
    print("%f %f" %(it[0], y[it.index]))
    it.iternext()
