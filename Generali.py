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
INSEE_data = pd.read_excel("MDB-INSEE-V2.xls")

#%%#
Code_Postale = pd.read_csv('correspondance-code-insee-code-postal.csv', error_bad_lines=False , sep = ';')


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


#%% 
#Drop Unamed and Identifiant and superficief and Insee
train_total[train_total['Insee'].isna()].target.value_counts(dropna=False)

#%%
train_total[train_total['Insee'].isna()].superficief.value_counts(dropna=False)
train_total = train_total.dropna()


#%%

Code_Postale = Code_Postale[['Code INSEE' , 'Code Postal']]
train_code = pd.merge(Code_Postale , train_total, how  = 'right', left_on='Code INSEE', right_on='Insee')


#%%

#%%
Insee_columns = ['CODGEO' , 'Orientation Economique' , 'Indice Démographique' , 'Population' , 'Nb Résidences Principales', 'Score Croissance Population'  , 'SEG Environnement Démographique Obsolète' , 'Nb Actifs Non Salariés']

#%%
train_Insee = pd.merge(INSEE_data[Insee_columns] , train_total, how  = 'right', left_on='CODGEO', right_on='Insee')

#%% 
train_Insee = train_Insee.drop(columns = ['Unnamed: 0_y' , 'superficief' , 'Unnamed: 0_x' , 'Identifiant' ] , axis =1)

#%%
train_Insee.isna().sum()


#%%
from xgboost import XGBClassifier

alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)



#%%
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [6, 8, 10, 12],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier( class_weight= 'balanced' ) 
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

#%%

X_train = train_Insee.drop(columns = ['Insee'  , 'CODGEO' , 'Orientation Economique' , 'SEG Environnement Démographique Obsolète'] , axis = 1)
X_train['EXPO'] = pd.to_numeric(X_train['EXPO'] , errors='coerce')
X_train['ft_22_categ'] = X_train['ft_22_categ'].fillna(X_train['ft_22_categ'].mean())
X_train['EXPO'] = X_train['EXPO'].fillna(X_train['EXPO'].mean())
X_train = X_train.dropna(subset = ['superficief_enc' , 'Population'])
y_train = X_train['target']
X_train = X_train.drop(columns = 'target' , axis =1)
#%%
grid_search.fit(X_train, y_train)

#%%
grid_search.best_params_

#%%
#

#grid_search.best_params_grid_
#{'bootstrap': True,
 #'max_depth': 12,
 #'max_features': 3,
 #'min_samples_leaf': 3,
 #'min_samples_split': 8,
 #'n_estimators': 1000}


#%%
clf = RandomForestClassifier(max_depth=12 , max_features=3 , min_samples_leaf=3 , min_samples_split=8 , bootstrap= True  , n_estimators=1000 , class_weight= 'balanced')

#%%
clf.fit(X_train , y_train)

#%%
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


pred_cv_label = cross_val_predict(clf, X_train, np.ravel(y_train),scoring=gini_sklearn,
                             cv=10, n_jobs=-1)

#%%
def ginic(actual, pred):
    actual = np.asarray(actual) #In case, someone passes Series or list
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_s.sum() - (n + 1) / 2.0
    return giniSum / n
 
def gini_normalizedc(a, p):
    if p.ndim == 2:#Required for sklearn wrapper
        p = p[:,1] #If proba array contains proba for both 0 and 1 classes, just pick class 1
    return ginic(a, p) / ginic(a, a)

gini_sklearn = metrics.make_scorer(gini_normalizedc, True, True)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

cv_1 = StratifiedKFold(n_splits=5, random_state=1).split(X_train, y_train)

#Check cross validation scores
cross_val_score(clf, X_train, y_train, cv=cv_1, scoring=gini_sklearn, verbose=1, n_jobs=-1)

#%%


test_X  = pd.read_csv('X_test.csv' )



#%%
test_total = MultiColumnLabelEncoder(columns = Label_encoder_columns).fit_transform(train_X)


#%%

test_total['superficief_enc'] = train_total['superficief_enc'].fit_transform(test_X['superficief'])
#test_total['Insee_enc'] = target_encoder.fit_transform(test['Insee'])


test_Insee = pd.merge(INSEE_data[Insee_columns] , test_total, how  = 'right', left_on='CODGEO', right_on='Insee')
#https://medium.com/@pouryaayria/k-fold-target-encoding-dfe9a594874b
#%% 
test_Insee = test_Insee.drop(columns = ['Unnamed: 0_y' , 'superficief' , 'Unnamed: 0_x' , 'Identifiant' ] , axis =1)

#%% 
X_test = test_Insee.drop(columns = ['Insee'  , 'CODGEO' , 'Orientation Economique' , 'SEG Environnement Démographique Obsolète'] , axis = 1)
X_test['EXPO'] = pd.to_numeric(X_test['EXPO'] , errors='coerce')
X_test['ft_22_categ'] = X_test['ft_22_categ'].fillna(X_test['ft_22_categ'].mean())
X_test['EXPO'] = X_test['EXPO'].fillna(X_test['EXPO'].mean())
X_test = X_test.dropna(subset = ['superficief_enc' , 'Population'])

