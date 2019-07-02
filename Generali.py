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
import pandas_profiling


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

gini_sklearn = metrics.make_scorer(gini_normalized, True, True)


#%%#
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

def myfillna(series):
    if series.dtype is pd.np.dtype(float):
        return series.fillna(0)
    elif series.dtype is pd.np.dtype(object):
        return series.fillna('missing')
    else:
        return series




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

#%%#
INSEE_data = pd.read_excel("MDB-INSEE-V2.xls")

#%%#
Code_Postale = pd.read_csv('correspondance-code-insee-code-postal.csv', error_bad_lines=False , sep = ';')


#%%#
chomage  = pd.read_excel("sl_cho_2019T1.xls" , sheet_name = "Département" , header = 3)
chomageT1 = chomage[['Code', 'T1_2019']]
#%%#
flux_mobilite  = pd.read_excel("base-excel-flux-mobilite-residentielle-2016.xls" , header = 5)


#%%#

train_X = pd.read_csv('X_train.csv' )
train_y = pd.read_csv('y_train_saegPGl.csv' )

#%%#

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
train_X['Code'] = train_X['Insee'].str[0:2]

#%%
Label_encoder_columns = [ 'ft_5_categ', 'ft_6_categ', 'ft_7_categ', 'ft_8_categ', 'ft_9_categ',
       'ft_10_categ', 'ft_11_categ', 'ft_12_categ', 'ft_13_categ',
       'ft_14_categ', 'ft_15_categ', 'ft_16_categ', 'ft_17_categ',
       'ft_18_categ', 'ft_19_categ',   'ft_23_categ', 'ft_24_categ']

train_Label_Encoder = MultiColumnLabelEncoder(columns = Label_encoder_columns).fit_transform(train_X)

#%%
train_total  = pd.merge(train_Label_Encoder, train_y, on=['Identifiant', 'Identifiant'])



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
train_total[train_total['Insee'].isna()].superficief.value_counts(dropna=False)

#%%
#Insee and superficief are linked with NAN values , we can drop all the missing values from there and look at the 
train_total = train_total.dropna(subset = ['Insee' , 'superficief'])
train_total['ft_22_categ'] = train_total['ft_22_categ'].fillna(train_total['ft_22_categ'].median())

#%%
train_total['EXPO'] = train_total['EXPO'].str.replace("," ,".")
train_total['EXPO'] = pd.to_numeric(train_total['EXPO'])

#%%
train_total = train_total.drop(columns = ['Identifiant' , 'Unnamed: 0_x'  , 'Unnamed: 0_y'] , axis =1)



#%%

X_train_xgb = train_total.drop(columns = ['target' ,  'Insee' , 'Code'] , axis = 1)
y_train_xgb = train_total['target']


alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)





#%%

cv_1 = StratifiedKFold(n_splits=10, random_state=1).split(X_train, y_train)

#Check cross validation scores
cross_val_score(alg, X_train_xgb, y_train_xgb, cv=cv_1, scoring=gini_sklearn,  verbose=1, n_jobs=-1)




#%% [markdown]
# Now we can add information about the INSEE code by geographical and economical point of view
# CHOMAGE

train_Chomage = pd.merge(chomageT1, train_total, how  = 'right', left_on='Code', right_on='Code')


train_Mobilité = pd.merge( flux_mobilite , train_Chomage ,  how  = 'right', left_on='CODGEO', right_on='Insee')

#%%
#Insee_columns = ['CODGEO' , 'Orientation Economique' , 'Indice Démographique' , 'Population' , 'Nb Résidences Principales', 'Score Croissance Population'  , 'SEG Environnement Démographique Obsolète' , 'Nb Actifs Non Salariés']
#train_Insee = pd.merge(INSEE_data[Insee_columns] , train_total, how  = 'right', left_on='CODGEO', right_on='Insee')



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
                          cv = 3, n_jobs = -1, verbose = 2 , scoring = gini_sklearn)

#%%

X_train_rf = train_Mobilité.drop(columns = ['LIBGEO', 'Insee' , 'Code', 'target' , 'CODGEO'] , axis = 1)
X_train_rf = X_train_rf.fillna(0)
y_train_rf = train_Mobilité['target']

#%%
grid_search.fit(X_train_rf, y_train_rf)

#%%
grid_search.best_params_



#%%
clf = RandomForestClassifier(max_depth=6 , max_features=3 , min_samples_leaf=3 , min_samples_split=8 , bootstrap= True  , n_estimators=300)





#%%

cv_1 = StratifiedKFold(n_splits=10, random_state=1).split(X_train_rf, y_train_rf)
#%%
#Check cross validation scores
cross_val_score(clf, X_train_rf, y_train_rf, cv=cv_1, scoring=gini_sklearn,  verbose=1, n_jobs=-1)
#%%
cross_val_score(alg, X_train_rf, y_train_rf, cv=cv_1, scoring=gini_sklearn,  verbose=1, n_jobs=-1)


#%%
test_X  = pd.read_csv('X_test.csv' )



#%%
test_total = MultiColumnLabelEncoder(columns = Label_encoder_columns).fit_transform(test_X)


#%%

#test_total['superficief_enc'] = train_total['superficief_enc'].fit_transform(test_X['superficief'])
#test_total['Insee_enc'] = target_encoder.fit_transform(test['Insee'])

test_total[test_total['Insee'].isna()].superficief.value_counts(dropna=False)

#%%
#Insee and superficief are linked with NAN values , we can drop all the missing values from there and look at the 
test_total = test_total.dropna(subset = ['Insee' , 'superficief'])
test_total['ft_22_categ'] = test_total['ft_22_categ'].fillna(test_total['ft_22_categ'].median())

#%%
test_total['EXPO'] = test_total['EXPO'].str.replace("," ,".")
test_total['EXPO'] = pd.to_numeric(test_total['EXPO'])

#%%
test_total = test_total.drop(columns = ['Identifiant' , 'Unnamed: 0'  , 'Insee'] , axis =1)


#%%


y_pred = alg.predict(test_total)
#%%
output=pd.DataFrame(data={"target":y_pred})

output.to_csv('prediction_xgb.csv' , index = False)

#%%

ziad = pd.read_excel("prediction_xgb_ziad.xls" , delimiter= ";" , header =0)

#%%
ziad.to_csv('prediction_xgb_ziad2.csv'  , sep ="," , index = False)

#%%
ziad1 = pd.read_csv("prediction_xgb_ziad1.csv" , delimiter= "," , index_col= 0)

#%%
