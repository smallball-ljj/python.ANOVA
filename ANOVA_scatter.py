# ==================================================
rootpath=r'E:\ljj\aa\demo' # your saopy file path
import sys
sys.path.append(rootpath) # you can directly import the modules in this folder
sys.path.append(rootpath+r'\saopy\surrogate_model')
sys.path.append(rootpath+r'\saopy')
# ==================================================
from saopy.sampling_plan import *
from saopy.surrogate_model.ANN import *
from saopy.surrogate_model.surrogate_model import *

import numpy as np
import matplotlib.pyplot as plt
import csv



def generate_X_final(number_of_points_main_effect,number_DOE,lower_bound,upper_bound):
    for i in range(len(lower_bound)):
        sp = random_lhs(number_DOE, len(lower_bound)-1)
        sp.begin_sampling()
        lower_bound_new=lower_bound[0:i]+lower_bound[i+1:len(lower_bound)]
        upper_bound_new=upper_bound[0:i]+upper_bound[i+1:len(lower_bound)]
        sp.inverse_norm(lower_bound_new, upper_bound_new)

        x_main_effect = np.linspace(lower_bound[i], upper_bound[i], number_of_points_main_effect) #scatter points to calculate main effect

        X_new=np.zeros((number_of_points_main_effect*number_DOE,len(lower_bound)))
        for j in range(number_of_points_main_effect):
            X_new[j*number_DOE:(j+1)*number_DOE,i]=x_main_effect[j]
            X_new[j * number_DOE:(j + 1) * number_DOE, 0:i]=sp.X[:,0:i]
            X_new[j * number_DOE:(j + 1) * number_DOE, i+1:len(lower_bound)] = sp.X[:, i:len(lower_bound)-1]

        try:
            X_final=np.vstack((X_final,X_new))
        except:
            X_final=X_new

    np.savetxt('X_final.csv', X_final, delimiter=',')


def generate_y_final(surro_name):
    surro = load_obj(surro_name)
    X=np.loadtxt('X_final.csv',delimiter=',')
    y_final=surro.calculate(X)
    np.savetxt('y_final.csv', y_final, delimiter=',')


def generate_data():
    X = np.loadtxt('X_final.csv', delimiter=',')
    y = np.loadtxt('y_final.csv', delimiter=',')
    y.resize((y.shape[0], 1))
    data=np.hstack((X,y))
    np.savetxt('data_final.csv', data, delimiter=',')


def ANOVA():
    data=np.loadtxt('data_final.csv', delimiter=',')
    dimension=data.shape[1]-1

    miu_total=data[:,-1].mean()
    sigma_total=((data[:,-1]-miu_total)**2).mean()

    variance_i=np.zeros((dimension,1))
    for d in range(dimension):
        tmp=[] # record all the different value of one varible
        for i in range(data.shape[0]):
            flag = 1
            for j in tmp:
                if data[i,d]==j:
                    flag=0
                    break
            if flag==1:
                tmp.append(data[i,d])

        miu_i_first_term_average=np.zeros((len(tmp),1))
        for j in range(len(tmp)):
            miu_i_first_term_total = []
            for i in range(data.shape[0]):
                if data[i,d]==tmp[j]:
                    miu_i_first_term_total.append(data[i,-1])
            miu_i_first_term_average[j]=np.array(miu_i_first_term_total).mean()

        miu_i=miu_i_first_term_average-miu_total

        variance_i[d]=(miu_i**2).mean()/sigma_total

    np.savetxt('variance_i.csv', variance_i, delimiter=',')
    variance_i_list=[]
    for i in range(dimension):
        variance_i_list.append(variance_i[i,0])
    x_label=[]
    for i in range(dimension):
        x_label.append('x'+str(i))
    plt.bar(x_label,variance_i_list,width=0.2)
    plt.ylabel('Variance')
    plt.show()


if __name__ == '__main__':
    lower_bound = [0.2,0.1,0.1,0.1,0.15,0.05,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0,10,0]
    upper_bound = [0.7,0.5,0.9,0.9,0.7,0.15,0.9,0.9,0.9,0.9,0.9,0.9,1,0.99,1,30,1]

    number_of_points_main_effect=10
    number_DOE = 300

    generate_X_final(number_of_points_main_effect, number_DOE, lower_bound, upper_bound)
    generate_y_final('best_surro')
    generate_data()
    ANOVA()