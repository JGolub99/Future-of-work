#Begin by importing all of the necessary libraries:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

#We must begin with some user input:

my_data = ["/Users/jacob/Documents/4YP data/ONET_18-10-21/Skills.xlsx"]


#Lets now begin defining our function!

#Import the data:

def import_data(data):
    dataframes = [] #List of dataframes
    for i in data:
        dataframes.append(pd.read_excel(i))
    return dataframes


#Clean the data:

def title_set(my_string):
    my_list = []
    my_list[:0] = my_string
    my_list.remove("_")
    my_list.append(".00")
    my_output = "".join(my_list)
    return my_output


def clean_data():
    pass

#Concatenate the data:

def concatenate_data():
    pass

#Generate training and test sets:

def create_sets():
    pass

#Standardise the datasets:

def standardise():
    pass

#Fit a GP classifier:

def fit_gp():
    pass

#Calculate AUC:

def calc_AUC(gpc,X_train,y_train):
    y_probas = cross_val_predict(gpc,X_train,y_train,cv=5,method="predict_proba")
    y_scores = y_probas[:,1]
    return roc_auc_score(y_train,y_scores)

#Permutation method:

def permute():
    pass

#Logarithm that can deal with zero:

def log_calc(my_list):    #This function deals with values of 0
    my_output = [0]*len(my_list)
    for i in range(len(my_list)):
        if my_list[i] != 0.0:
            my_output[i] = np.log(my_list[i])
    return my_output        
            
#Calculate entropy from dataframe:

def calc_entropy(gpc,X_test,y_test):
    r = permutation_importance(gpc,X_test,y_test,n_repeats=30,random_state=0)
    temp = abs(r.importances_mean)
    tempnew = temp/sum(temp)
    vector1 = np.array(tempnew)
    vector2 = np.array(log_calc(abs(r.importances_mean)))
    return -1*np.dot(vector1,vector2)

#Grid search method:

def grid_search():
    pass

#Print accuracy vs interpretability:

def plot_accvint_graph():
    pass


def main():
    my_dataframes = import_data(my_data)
    print(my_dataframes[0].head())


if __name__ == "__main__":
    main()