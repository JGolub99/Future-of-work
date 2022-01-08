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
from sklearn.cluster import KMeans

#We must begin with some user input:

#my_input_data = ["/Users/jacob/Documents/4YP data/ONET_18-10-21/Skills.xlsx","/Users/jacob/Documents/4YP data/ONET_18-10-21/Abilities.xlsx"] #Skills, knowledge, abilites etc. (all of the same form)
my_input_data = ["/Users/jacob/Documents/4YP data/ONET_18-10-21/Skills.xlsx"]
imp_lev = "Level"  # Importance or Level

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


def transpose_data(a,b):
    n_jobs = len(set((a["Title"])))
    n_variables = len(set((a["Element Name"])))
    
    for i in range(n_jobs):
        b[a["Element Name"][i]] = ""
    
    for j in range(n_jobs):
        x = a.loc[a["Title"] == b.iloc[j,1]]
        y = x["Data Value"]
        y.reset_index(drop = True, inplace = True)
        for i in range(n_variables):
            b[b.columns[2+i]][j] = y[i]
    
    return b


#Concatenating dataframes of variables size:

def concat_data(a,b,key):
    for i in list(b["O*NET-SOC Code"]):
        a.loc[a["O*NET-SOC Code"] == i,key] = list(b.loc[b["O*NET-SOC Code"] == i,key])

#NOTE - this concat method will add ONE column (key) from b to a, you will need to run this many times to do all columns

#Generate training and test sets:

def create_sets(data,key):
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(data,data[key]):
        a = data.loc[train_index]
        b = data.loc[test_index]
    a.reset_index(drop=True,inplace=True)
    b.reset_index(drop=True,inplace=True)
    return a,b

#Standardise the datasets:

def standardise(template,subliminary,columns):  #Template is the df we base the scaler on (training) and subliminary is what we apply it to
    scaler = StandardScaler(with_mean=True,with_std=True)
    scaler.fit(template.iloc[:,2:columns])
    scaled_training_values = scaler.transform(template.iloc[:,2:columns])
    scaled_test_values = scaler.transform(subliminary.iloc[:,2:columns])
    scaled_train_set = template.copy()
    scaled_test_set = subliminary.copy()
    
    #we create temporary data frames from the numpy arrays we've just created
    
    temporary = pd.DataFrame(data=scaled_training_values)
    temporary2 = pd.DataFrame(data=scaled_test_values)
    for i in range(2,columns):
        scaled_train_set[scaled_train_set.columns[i]] = temporary[temporary.columns[i-2]]
        scaled_test_set[scaled_test_set.columns[i]] = temporary2[temporary2.columns[i-2]]
    
    return scaled_train_set,scaled_test_set,scaler



#Calculate AUC:

def calc_AUC(gpc,X_train,y_train):
    y_probas = cross_val_predict(gpc,X_train,y_train,cv=5,method="predict_proba")
    y_scores = y_probas[:,1]
    return roc_auc_score(y_train,y_scores)


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

#Method for determining auto label from probability

def label_auto(my_input):
    if my_input - 0.5 < 0:
        return 0
    else:
        return 1

def main():
    
    #Calculate number of input dataframes:
    
    n_inputdata = len(my_input_data)
    
    #Begin by importing the data
    
    my_input_dataframes = import_data(my_input_data) # Import all relevant input data
    my_input_dataframes2 = [] # This one is important for transposing the dataframes
    my_output_dataframe = pd.read_excel("/Users/jacob/Documents/4YP data/ONET_18-10-21/US_data_email.xlsx") # Input output data
    
    #Clean the input data:
    #Drop either importance or level rows
    #Drop unimportant columns
    #Transpose dataframes using transpose_data()

    
    for i in range(n_inputdata):
        my_input_dataframes[i] = my_input_dataframes[i].loc[my_input_dataframes[i]["Scale Name"] == imp_lev]
        my_input_dataframes[i].reset_index(drop = True, inplace = True)
        my_input_dataframes[i] = my_input_dataframes[i].drop(columns = ["Scale ID","Scale Name","N","Recommend Suppress","Not Relevant","Date","Domain Source","Standard Error","Lower CI Bound","Upper CI Bound"])
    
        my_input_dataframes2.append(my_input_dataframes[i][["O*NET-SOC Code","Title"]])
        my_input_dataframes2[i].drop_duplicates(inplace=True)
        my_input_dataframes2[i].reset_index(drop = True, inplace = True)
        
        my_input_dataframes2[i] = transpose_data(my_input_dataframes[i],my_input_dataframes2[i])
        
    
    #Clean the output data
    
    my_output_dataframe = my_output_dataframe[["Occupation Name","BLS codes","Training set automatable labels"]] # Keep relevant columns
    my_output_dataframe.iloc[:,1] = my_output_dataframe.iloc[:,1].apply(title_set) # Change the IDs of the output soc codes
    
    
    
   #Lets now drop the repeated columns in our input dataframes and concatenate them into a single input dataframe:
    
    if n_inputdata > 1:
        for i in range(n_inputdata-1):
            my_list = list(my_input_dataframes2[i+1].columns)
            my_list.remove('O*NET-SOC Code')
            for j in my_list:
                concat_data(my_input_dataframes2[0],my_input_dataframes2[1+i],j)
        my_input_dataframe = my_input_dataframes2[0]
    else:
        my_input_dataframe = my_input_dataframes2[0]
    
    
    #Now add an auto-label value defaulted to NaN
    
    my_input_dataframe["Auto label value"] = np.nan
    
    #Lets now add the auto-values from output_dataframe to input_dataframe (we'll use a slightly different code to the concat_data method):
    
    for i in list(my_output_dataframe["BLS codes"]):
        my_input_dataframe.loc[my_input_dataframe["O*NET-SOC Code"] == i,"Auto label value"] = list(my_output_dataframe.loc[my_output_dataframe["BLS codes"] == i,"Training set automatable labels"])
    
    
    #Lets now drop the jobs with NaN values:
    
    training_set = my_input_dataframe.dropna(axis=0,how="any")
    training_set.reset_index(drop = True, inplace=True)
    training_set.info()
    n_columns = len(training_set.columns)
    
    #Lets now create some training and test sets:
    
    (strat_train_set,strat_test_set) = create_sets(training_set,"Auto label value")
    
    #And now we standardise:
    
    (scaled_train_set,scaled_test_set,scaler) = standardise(strat_train_set,strat_test_set,n_columns-1)
    
    
    #We now have a training and test set just the way we want it!
    
    n_rows = len(scaled_train_set.index)
    
    X = np.array([scaled_train_set.iloc[:,2:n_columns-1]])
    Y = np.array([scaled_train_set.iloc[:,n_columns-1]])
    X = np.reshape(X,(n_rows,n_columns-3)) #Reshape to go from 3d matrix to 2d
    Y = np.reshape(Y,(n_rows,1))
    
    
    X_train = np.array(scaled_train_set.iloc[:,2:n_columns-1])
    y_train = np.array(scaled_train_set.iloc[:,n_columns-1])
    X_test = np.array(scaled_test_set.iloc[:,2:n_columns-1])
    y_test = np.array(scaled_test_set.iloc[:,n_columns-1])
    
    #Gridsearch:
    
    n_lengthscale = 10
    n_const = 10  #Number of iterations we want to search through
    
    lengthscale = np.linspace(7.5,8.5,n_lengthscale)
    const = np.linspace(120.0,150.0,n_const)
    
    resultsdf = pd.DataFrame({'length_scale':[0.0]*(n_lengthscale*n_const),'const':[0.0]*(n_lengthscale*n_const),'AUC':[0.0]*(n_lengthscale*n_const),"log-likelihood":[0.0]*(n_lengthscale*n_const),"entropy":[0.0]*(n_lengthscale*n_const)})

    iteration = 0

    for i in lengthscale:
        for j in const:
            kernel = j*RBF(i)
            gpc = GaussianProcessClassifier(kernel=kernel,optimizer=None).fit(X, Y)
            resultsdf.iloc[iteration]['AUC'] = calc_AUC(gpc,X_train,y_train)
            resultsdf.iloc[iteration]['log-likelihood'] = gpc.log_marginal_likelihood(theta=None, eval_gradient=False, clone_kernel=True)
            resultsdf.iloc[iteration]['length_scale'] = i
            resultsdf.iloc[iteration]['const'] = j
            resultsdf.iloc[iteration]['entropy'] = calc_entropy(gpc,X_test,y_test)
            iteration += 1
            
    resultsdf = resultsdf.sort_values(by=['log-likelihood'])
    
    plt.scatter(resultsdf['entropy'],resultsdf['log-likelihood'])
    plt.xlabel("Feature entropy")
    plt.ylabel("log-likelihood")
    plt.savefig('clustergraph.png',bbox_inches='tight')
    plt.show()
    
    #User input required to identify the number of clusters
    
    n = input("How many clusters?")
    
    km = KMeans(n_clusters=int(n))
    c_predicted = km.fit_predict(resultsdf[["log-likelihood","entropy"]])
    resultsdf["cluster"]=c_predicted
    
    #Generate empty lists and dictionaries for the upcoming methods
    
    dataframes = []
    models = []
    tables = []
    importance_dict = {}
    colors = {0:"blue",1:"green",2:"red",3:"cyan",4:"yellow",5:"black",6:"magenta",7:"white"} #colour matrix
    
    
    
    for i in range(int(n)):
        
        #With clusters identified we can split the data into seperate dataframes for each cluster and plot them
        
        dataframes.append(resultsdf[resultsdf.cluster == i])
        plt.scatter(dataframes[i].entropy,dataframes[i]["log-likelihood"],color=colors[i])
        dataframes[i].sort_values(by=['log-likelihood'])
        
        #Now create a model for the modes of each cluster and fill the models list
        
        kernel_it = dataframes[i].iloc[0]['const']*RBF(dataframes[i].iloc[0]['length_scale'])
        gpc_it = GaussianProcessClassifier(kernel=kernel_it,optimizer=None).fit(X, Y)
        models.append(gpc_it)
        
        #We are now concerned with generating the interpretations for each cluster using the permutation method
        
        tables.append(pd.DataFrame({"Features":np.array(training_set.columns[2:n_columns-1])}))
        r = permutation_importance(models[i],X_test,y_test,n_repeats=30,random_state=0)
        tables[i]["Importances"] = abs(r.importances_mean)
        tables[i] = tables[i].sort_values(by=["Importances"],ascending=False)
        tables[i].reset_index(drop = True, inplace = True)
        
        q = tables[i]
        importance_dict["Importance{}".format(i)] = q["Features"]

    plt.xlabel("Feature entropy")
    plt.ylabel("log-likelihood")
    plt.show()

    #Make a dataframe from the interpretation dictionary
    
    importance_table = pd.DataFrame(data=importance_dict)
    print(importance_table)
    
    #Finally we want one final interpretation which takes all from the importance_table into account
    
    final_importance_table = pd.DataFrame({"Features":np.array(training_set.columns[2:n_columns-1]),"Importance":np.zeros(np.shape(training_set.columns[2]))})
    feature_list = final_importance_table['Features'].tolist()
    
    for i in feature_list:
        for j in range(int(n)):
            index_val = importance_table.index[importance_table["Importance{}".format(j)]==i]
            final_importance_table.loc[final_importance_table["Features"]==i,"Importance"] += index_val.tolist()[0]
    
    final_importance_table = final_importance_table.sort_values(by=["Importance"],ascending=True)
    final_importance_table.reset_index(drop = True, inplace = True)
    print(final_importance_table)
    
    #Lets divert our attention towards making predictions on our unknown jobs:
    
    unknown_jobs = my_input_dataframe[my_input_dataframe.isna().any(axis=1)]
    X = scaler.transform(unknown_jobs.iloc[:,2:n_columns-1])
    
    column_titles = []
    
    for i in range(int(n)):
        probs = models[i].predict_proba(X)
        unknown_jobs["Auto probability{}".format(i)] = probs[:,1]
        column_titles.append("Auto probability{}".format(i))
    
    unknown_jobs["Auto probability"] = unknown_jobs[column_titles].mean(axis=1)
    
    unknown_jobs["Auto label value"] = unknown_jobs["Auto probability"].apply(label_auto)
    print(unknown_jobs.head())
    unknown_jobs.to_excel("unknown_jobs.xlsx", index=False)
    final_importance_table.to_excel("importance_table.xlsx", index=False)




if __name__ == "__main__":
    main()