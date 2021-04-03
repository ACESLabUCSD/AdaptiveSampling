'''
Test functions for optimization, implemnented based on https://en.wikipedia.org/wiki/Test_functions_for_optimization
'''


import numpy as np


def rastrigin(x, A=10):
	x = np.asarray(x)
	f = A*len(x) + np.sum(x**2 - A*np.cos(2*np.pi*x))
	return f

def ackley(x):
	# only 2d
	x = np.asarray(x)
	f = -20*np.exp(-0.2*np.sqrt(0.5*np.sum(x**2))) - np.exp(0.5*np.sum(np.cos(2*np.pi*x))) + np.exp(1.) + 20
	return f

def sphere(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	x = np.asarray(x)
	f = np.sum(x**2)
	return f

def rosenbrock(x):
	f = 0
	for i in range(len(x)-1):
		f += 100*((x[i+1]-(x[i]**2))**2)+((1-x[i])**2)
	return f

def beale(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	f = (1.5-x[0]+(x[0]*x[1]))**2 + (2.25-x[0]+(x[0]*(x[1]**2)))**2 + (2.625-x[0]+(x[0]*(x[1]**3)))**2
	return f

def himmelblau(x):
	# only 2d
	assert len(x)==2, 'only 2d input vectors accepted for this function'
	f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
	return f
