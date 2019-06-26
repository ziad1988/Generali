#%%#Challenge goals [markdown]
The goal of the challenge is to predict if a building will have an insurance claim during a certain period. You will have to predict a probability of having at least one claim over the insured period of a building. The model will be based on the building characteristics. The target variable is a:

-	1 if the building has at least a claim over the insured period.
-	0 if the building doesn’t have a claim over the insured period.
During this challenge, you are encouraged to use external data. For instance: shops number by INSEE code (geographical code), unemployment rate by INSEE code, weather…

Some data can be found on the following website: data.gouv.fr
#%%#
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.svm import SVC

#%%#
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier


#%%#
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)




#%%#

train_X = pd.read_csv('X_train.csv' )
train_y = pd.read_csv('y_train_saegPGl.csv' )

#%%#
import pandas_profiling
pandas_profiling.ProfileReport(train_X)

#%%
import matplotlib.pyplot as plt
# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(train_y['target'])
# set title and labels
ax.set_title('Target Distribution')
ax.set_xlabel('Points')
ax.set_ylabel('Frequency')




#%%
data = []
for f in train_X.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif train_X[f].dtype == float:
        level = 'interval'
    elif train_X[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = train_X[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
    
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)





#%%
def target_encoder(df, column, target, index=None, method='mean'):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. Main purpose is to deal
    with high cardinality categorical features without exploding dimensionality. This replaces the categorical variable
    with just one new numerical variable. Each category or level of the categorical variable is represented by a
    summary statistic of the target for that level.
    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (str): Categorical variable column to be encoded.
        target (str): Target on which to encode.
        index (arr): Can be supplied to use targets only from the train index. Avoids data leakage from the test fold
        method (str): Summary statistic of the target. Mean, median or std. deviation.
    Returns:
        arr: Encoded categorical column.
    """

    index = df.index if index is None else index # Encode the entire input df if no specific indices is supplied

    if method == 'mean':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].mean())
    elif method == 'median':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].median())
    elif method == 'std':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].std())
    else:
        raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(method))

    return encoded_column




#%%

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


#%%

def myfillna(series):
    if series.dtype is pd.np.dtype(float):
        return series.fillna(0)
    elif series.dtype is pd.np.dtype(object):
        return series.fillna('missing')
    else:
        return series


#%%
Lesion_fill = Lesion1.apply(myfillna)
Lesion_fill.isna().sum()


#%%
Label_encoder_columns = [ 'ft_5_categ', 'ft_6_categ', 'ft_7_categ', 'ft_8_categ', 'ft_9_categ',
       'ft_10_categ', 'ft_11_categ', 'ft_12_categ', 'ft_13_categ',
       'ft_14_categ', 'ft_15_categ', 'ft_16_categ', 'ft_17_categ',
       'ft_18_categ', 'ft_19_categ',   'ft_23_categ', 'ft_24_categ']

train_Label_Encoder = MultiColumnLabelEncoder(columns = Label_encoder_columns).fit_transform(train_X)

#%%
train_total  = pd.merge(train_Label_Encoder, train_y, on=['Identifiant', 'Identifiant'])


#%%

train_total['superficief_enc'] = target_encoder(train_total , 'superficief' , 'target')
train_total['Insee_enc'] = target_encoder(train_total , 'Insee' , 'target')



#%% [markdown]
# 5  Missing variables 
#1. Superficief
#2. ft_22_categ
#3. Insee
#4.superficie_enc
#5.Insee_enc