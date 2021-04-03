from __future__ import print_function

import os
import argparse
import time
import collections
import pickle
import copy
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


from utils.sampling_utils import Genetic_sampler, Gaussian_sampler, Zoom_sampler
import utils.example_functions as example_functions

# define your objective function here
def evaluator_fn(x):
	'''
		x: <d> the hyperparameter vector
		returns: value of the objective function at \vec x 
	'''
	# example: return np.cos(4*np.pi*x[0]) + np.cos(4*np.pi*x[1]) - 5*(x[0]+x[1]) + 2
	f = np.absolute(x[0]-x[1]) + ((x[0]+x[1]-1)/3)**2

	return f


def get_run_config():
	parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')

	parser.add_argument('--name', default='test_fn_0',
								help='name of the experiment (default: test_fn_0)')
	parser.add_argument('--minimize', action='store_true',
								help='if selected, the function will be minimized, otherwise maximized')
	parser.add_argument('--test_fn', type=str,
								help='(optional) choose from common optimization test functions [rastrigin, ]')
	parser.add_argument('--plot_contour', action='store_true',
								help='if selected, the sampler will save contours of the objective function along with per-iteration samples')
	parser.add_argument('--seed', default=0, type=int,
								help='random seed (default: 0)')
	
	#------------- Sampler parameters
	parser.add_argument('--num_samples', default=50, type=int,
								help='per-iteration sample size (default: 50)')
	parser.add_argument('--dim', type=int,
								help='dimensionality of the search-space (default: None)')
	parser.add_argument('--path_to_boundaries', default='', type=str,
								help='path to csv file containing search-space boundaries (default: '')')
	parser.add_argument('--n_iter', default=50, type=int,
								help='number of optimization iterations (default: 50)')
	parser.add_argument('--n_parallel', default=1, type=int,
								help='number of cores for parallel evaluations (default:1)')
	parser.add_argument('--alpha_max', default=1.0, type=float,
								help='alpha_max parameter (default:1.0)')
	parser.add_argument('--early_stopping', default=10, type=int,
								help='number of iterations without improvement to activate early stopping (default: 10)')
	
	#-------------- Gaussian Sampler
	parser.add_argument('--u_random_portion', default=0.2, type=float,
								help='portion of samples to take unifromly random from the entire space (default:0.2)')
	parser.add_argument('--local_portion', default=0.4, type=float,
								help='portion of samples to take from gaussians using the Local method (default:0.4)')
	parser.add_argument('--cross_portion', default=0.4, type=float,
								help='portion of samples to take from gaussians using the Cross method (default:0.4)')
	parser.add_argument('--pair_selection', default='top_and_random', type=str,
								help='how to select sample pairs for crossing, choose from [random,top_scores,top_and_nearest,top_and_furthest,top_and_random] (default:top_and_random)')
	
	#-------------- Genetic Sampler
	parser.add_argument('--p_cross', default=0.8, type=float,
								help='probability of crossover (default: 0.8)')
	parser.add_argument('--p_swap', default=0.2, type=float,
								help='per-bit exchange probability (default: 0.2)')
	parser.add_argument('--p_mutate', default=0.8, type=float,
								help='probability of mutate (default: 0.8)')
	parser.add_argument('--p_tweak', default=0.05, type=float,
								help='per-bit tweaking probability (default: 0.05)')
	parser.add_argument('--mutate_scale', default=0.2, type=float,
								help='std of the noise added during mutation (default: 0.2)')
	
	args = parser.parse_args()
	return args

def main():
	args = get_run_config()
	np.random.seed(args.seed)

	# preparing the score_fn
	if args.test_fn is not None:
		evaluator = example_functions.__dict__[args.test_fn]
		args.name = args.test_fn
	else:
		evaluator = evaluator_fn

	# determining the search-space boundaries
	if len(args.path_to_boundaries)==0:
		assert args.dim is not None, 'Please either provide the search-space boundaries or the dimensionality of the search-space'
		boundaries = [[0, 1] for _ in range(args.dim)]
		print('=> no boundaries provided, setting default to [0, 1] for all dimensions')
	else:
		boundaries = np.genfromtxt(args.path_to_boundaries, delimiter=',')
		if args.dim is None:
			args.dim = len(boundaries)
		else:
			boundaries = boundaries[:args.dim, :]
		print('=> loaded boundaries are: \n', boundaries)
		
	args.save_path = os.path.join('artifacts', args.name)
	if not os.path.exists(args.save_path):
		os.makedirs(args.save_path)
	
	# Instantiating the sampler
	'''
	Examples:
		sampler = Genetic_sampler(boundaries, args.num_samples, args.p_cross, args.p_swap, args.p_mutate, args.p_tweak, args.mutate_scale)
		sampler = Gaussian_sampler(boundaries, args.num_samples, args.u_random_portion, args.local_portion, args.cross_portion, args.pair_selection)
		sampler = Zoom_sampler(boundaries, args.num_samples)
	'''
	
	sampler = Gaussian_sampler(boundaries, args.num_samples, args.u_random_portion, args.local_portion, args.cross_portion, args.pair_selection)
	
	print('=> Starting optimization')
	best_sample = sampler.run_sampling(evaluator, args.num_samples, args.n_iter, args.minimize, args.alpha_max, early_stopping=np.Infinity, 
		save_path=args.save_path, n_parallel=args.n_parallel, plot_contour=args.plot_contour, executor=mp.Pool)
	print('=> optimal hyperparameters:', best_sample)


if __name__ == '__main__':
	main()