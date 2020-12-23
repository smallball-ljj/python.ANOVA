# ==================================================
rootpath=r'F:\demo' # your saopy file path
import sys
sys.path.append(rootpath) # you can directly import the modules in this folder
sys.path.append(rootpath+r'\saopy\surrogate_model')
sys.path.append(rootpath+r'\saopy')
# ==================================================

from saopy.sampling_plan import *
from saopy.function_evaluation.benchmark_func import *
from saopy.surrogate_model.ANN import *
from saopy.surrogate_model.surrogate_model import *
from saopy.database import *

import numpy as np
import time


if __name__ == '__main__':

    lower_bound = [0, 0]
    upper_bound = [1, 1]

    print('initial sampling plan')
    number = 40; dimension = 2
    sp = optimal_lhs(number, dimension)
    sp.begin_sampling(population=30, iterations=30)  # the larger the better, but will take long time
    sp.inverse_norm(lower_bound, upper_bound)
    sp.output('X_new.csv')

    print('function evaluation')
    f = sphere(2)
    X = f.read_csv_to_np('X_new.csv')
    f.calculate(X)
    f.output('y_new.csv')
    print('database')
    stack('X.csv','X_new.csv')
    stack('y.csv','y_new.csv')

    surro=ANN(num_layers=2, num_neurons=50)
    surro.load_data(lower_bound, upper_bound)
    surro.normalize_all()

    surro.train(surro.normalized_X,surro.normalized_y)
    save_obj(surro,'best_surro') # save best model
