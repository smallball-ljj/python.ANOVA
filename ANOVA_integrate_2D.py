# ==================================================
rootpath = r'F:\demo'  # your saopy file path
import sys

sys.path.append(rootpath)  # you can directly import the modules in this folder
sys.path.append(rootpath + r'\saopy\surrogate_model')
sys.path.append(rootpath + r'\saopy')
# ==================================================
from saopy.surrogate_model.ANN import *
from saopy.surrogate_model.surrogate_model import *

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt

# the design variable X should be noramlized to [0,1] first
# the following example is two varibles funciton.

lower_bound = [0, 0]
upper_bound = [1, 1]

dimension=2
number_of_points_main_effect=10 # number of scatter points to calculate main effect

# best_surro = load_obj('best_surro')
# best_surro.plot(lower_bound, upper_bound, outer_iter=0, sample_flag=1)  # plot response surface
# def f(x1,x2): #surrogate model
#     X=np.array([[x1, x2]])
#     res=best_surro.calculate(X)
#     return res[0,0]


def f(x1,x2): #real function
    res=x1**2+x2**2
    return res

miu_total = integrate.nquad(f, [[0,1]]*dimension,opts=[{'epsabs':1e-1,'epsrel':1e-1}]*dimension)[0]
print('miu_total=',miu_total)

#note: the variance due to the design variable xi is comparable no matter it is divided by sigma_square or not.
# so it is not necessary to calculate it
# def f_sigma_square(x1,x2):
#     y=f(x1, x2)
#     res=(y-miu_total)**2
#     return res
# sigma_square = integrate.nquad(f_sigma_square, [[0,1]]*dimension)[0]
# print(sigma_square)


# calculate the main effect of variable xi
x_main_effect = np.linspace(0, 1, number_of_points_main_effect) #scatter points to calculate main effect
miu_i=np.zeros((dimension,number_of_points_main_effect))
for i in range(dimension):
    for j in range(number_of_points_main_effect):
        def f_main_effect(x1): # the number of variable in this function is one less than dimension
            X=[x1]
            X.insert(i, x_main_effect[j])
            y = f(X[0], X[1])
            return y
        miu_i[i,j] = integrate.nquad(f_main_effect, [[0, 1]] * (dimension-1))[0]-miu_total
print(miu_i)

variance_i=[]
for i in range(dimension):
    variance_i.append(integrate.trapz(miu_i[i]**2, x_main_effect))
print(variance_i)

plt.bar(['x1','x2'],variance_i,width=0.2)
plt.ylabel('Variance')
plt.show()