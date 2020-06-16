
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC, SVR
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler
import pandas as pd
import statistics, random
import sys 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

#from sklearn.linear_model import ridge
from sklearn.kernel_ridge import KernelRidge

from sklearn.tree import DecisionTreeRegressor

#dataset = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')
dataset = pd.read_csv(sys.argv[1])
df = pd.DataFrame(dataset)
test_labels = df['W51']
labels = df['W50']
#df = df.drop(columns=['W51'], axis=1)
#df = df.drop(columns = ['Product_Code'], axis=1)
product_codes = df['Product_Code']
df = df.drop(columns = ['Product_Code'], axis=1)
df = df.drop(df.loc[:, 'MIN':'Normalized 51'], axis=1)
data = df

##### Cross validation 

#alphas = [0.00001, 0.001, 0.001, 0.01, 0.1 , 1, 10, 100, 1000]
alpha = [0.1]
start = 0
alphas = [1]#x for x in range(1, 50-start+1, 1)]
#alphas = [50]

for alpha in alphas:
    true_vals = []
    mse_list = []
    predictions = []
    #X_train = []
    #X_test = []
    #y_train = []
    #y_test= []
    
    for i in range(0, data.shape[0]):
        X = data.loc[i,:]
        window_size = 51
        #window_size = len(X)-alpha-1
        size = window_size
        train, test = X[0:size], X[size:len(X)]
        
        avg_loss = 0
        history = [x for x in train]
        time = [ [ x+1] for x in range(0, window_size, 1)]
        #for j in range(alpha-1, len(X)-window_size-1):
         #   X_train.append(X[j:window_size+j])
          #  y_train.append(X[j+window_size])
        
        #X_test.append(X[len(X) - window_size - 1: len(X)-1])
        #y_test.append(X[len(X)-1])
        
        for t in range(len(test)):
           # print(history)
            
############ Running ARIMA
#            model = ARIMA(history, order = (5,1,0))
#            model_fit= model.fit(disp=0)
#            output = model_fit.forecast()
#            yhat = output[0]
            
######## Running regression meathods
            #clf = LinearRegression().fit(time, history)
            #clf = Ridge(alpha=alpha).fit(time,history)
            #clf = KernelRidge(alpha=alpha, kernel="poly", degree=2).fit(time,history)
            clf = KernelRidge(alpha=0.1, kernel="rbf", gamma=0.1).fit(time,history)
            
            #clf = SVR(kernel="linear", C=alpha,epsilon=0.0001).fit(time,history)
            #clf = SVR(kernel="poly",degree=2, C=alpha,epsilon=0.01).fit(time,history)
            #clf = SVR(kernel="rbf",gamma=0.0001, C=alpha,epsilon=0.00001).fit(time,history)
            
            #clf = DecisionTreeRegressor(max_depth=alpha).fit(time,history)
            
            #clf = MLPClassifier(hidden_layer_sizes=(100,), batch_size= 10, alpha=alpha).fit(time,history)
            #clf = MLPClassifier(hidden_layer_sizes=(100,), batch_size= 20, alpha=alpha, max_iter= 10000).fit(time,history)
            
            #avg_loss += clf.loss_
            yhat = clf.predict([[window_size+t]])
            
            predictions.append(yhat)
            obs = test[t]
        #history.pop(0)
        #history.append(obs)
            true_vals.append(obs)
            print(" ", i, " %f" % (yhat))
    
    error = mean_squared_error(true_vals, predictions)
    #error = r2_score(test,predictions)
    #print('for alpha=', alpha,' Test MSE: %.3f' % error)
    mse_list.append(error)

print(mse_list)
#plt.plot(test)
#plt.plot(predictions, color = 'blue')
#plt.show()



