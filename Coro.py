

#%%
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
import matplotlib.pyplot as plt




#%%
import os
cwd = os.getcwd()
cwd
os.chdir('C:\\Users\\znader\\Desktop\\Coronography')

#%%
coro = pd.read_excel("Ziad.xlsx" , header = 2 , index_col = None)


#%% [mardown]

#Missing Completely at Random, MCAR, means there is no relationship between the missingness of the data and any values, observed or missing. Those missing data points are a random subset of the data. There is nothing systematic going on that makes some data more likely to be missing than others.

#Missing at Random, MAR, means there is a systematic relationship between the propensity of missing values and the observed data, but not the missing data.

#Whether an observation is missing has nothing to do with the missing values, but it does have to do with the values of an individual’s observed variables. 
#So, for example, if men are more likely to tell you their weight than women, weight is MAR.

#Missing Not at Random, MNAR, means there is a relationship between the propensity of a value to be missing and its values. This is a case where the people with the lowest education are missing on education or the sickest people are most likely to drop out of the study.

#%%
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
coro_numeric = coro.select_dtypes(include=numerics)

#%%
#Treating NA values for numerical data
coro_numeric.isnull().sum()
#%%
coro_numeric['FEVG ventriculographie'].value_counts(dropna = False)
coro_numeric['Syntax_score_global'].value_counts(dropna = False)
coro_numeric['FFR'].value_counts(dropna = False)

#%%

# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(coro_numeric['Âge'])
# set title and labels
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')

#%%
fig, ax = plt.subplots()
ax.hist(coro_numeric['Poids'])
# set title and labels
ax.set_title('Poids Distribution')
ax.set_xlabel('Poids')
ax.set_ylabel('Frequency')



#%%
fig, ax = plt.subplots()
ax.hist(coro_numeric["Nb d'artère(s) dilatée(s)"])
# set title and labels
ax.set_title('Nb Artere Distribution')
ax.set_xlabel('Nb artere')
ax.set_ylabel('Frequency')

#%%
my_tab = pd.crosstab(index = coro["Abord artériel principal"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()

#%%
my_tab = pd.crosstab(index = coro["Abord artériel principal"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()

#%%
my_tab = pd.crosstab(index = coro["Survie H + 1 an"].fillna('missing'),  # Make a crosstab
                              columns="count"   )   # Name the count column

my_tab.plot.bar()


#%%

Survie_artère = pd.crosstab(index=coro["Survie H + 1 an"].fillna('missing'), 
                          columns=coro["Nb d'artère(s) dilatée(s)"])

Survie_artère

#%%

IDM_artère = pd.crosstab(index=coro['Infarctus du myocarde 1an'].fillna('missing'), 
                          columns=coro["Nb d'artère(s) dilatée(s)"])

IDM_artère


#%%

IDM_Site = pd.crosstab(index=coro['Infarctus du myocarde 1an'].fillna('missing'), 
                          columns=coro['Nb de site(s) dilaté(s)'])

IDM_Site

#%%

IDM_Site = pd.crosstab(index=coro['Infarctus du myocarde 1an'].fillna('missing'), 
                          columns=coro['Nb de site(s) dilaté(s)'])

IDM_Site



#%%
#Description of Numeric variables
coro_numeric.describe()



#%%
coro[coro['Taille'].isna() &  coro['Poids'].isna() & coro['IMC'].isna() ]
#Input can be replaced by mean , mean of the age and IMC can be calculated afterwards 



#%%
# 8 common values where all of the variables
coro[coro['Quantité de contraste utilisé (mL)'].isna() &  coro['Temps de scopie (min)'].isna() & coro['PDS (cGy x m²)'].isna() & coro['Air Kerma cumulé'].isna()]
coro[coro['Quantité de contraste utilisé (mL)'].isna()  & coro['PDS (cGy x m²)'].isna() & coro['Air Kerma cumulé'].isna()]


#%%
#FEVG Ventriculographie  : NA = potentiellement pas de ventriculographie
coro['FEVG ventriculographie'].value_counts(dropna = False)
coro['FEVG ventriculographie'].notna().sum()

#%%
#Longueur de Stent par Artère : 
coro['Longueur total de stent par artère'].value_counts(dropna = False)




#%%
# Variable to Remove
coro['Occlusion chronique.5'].value_counts(dropna = False)
coro['Occlusion chronique.1'].value_counts(dropna = False)

#%%
coro['Site précis.6'].value_counts(dropna = False)



#%%


## Handle the categorical ordinal variables
input_lesion = coro.iloc[: , 71 : 239]
input_lesion.describe()
input_lesion.iloc[:,0].value_counts(dropna = False)

#%%
Description_Lesion = pd.crosstab(index=input_lesion.iloc[:,0], 
                          columns=input_lesion.iloc[:,1])

Description_Lesion

#%%
Description_Lesion.plot(kind="bar", 
                 figsize=(8,8),
                 stacked=True)


#%%
input_lesion.isna().sum()


#%%

input_lesion.columns



#%%
input_lesion['Type de lésion'].value_counts(dropna = False)


#%%
input_lesion['Description stent restenosé'].value_counts(dropna = False)

#%%
input_lesion['Description stent restenosé'].value_counts(dropna = False)





#%%
input_lesion.isna().sum()


#%%
input_lesion['Description de la lésion 2'].value_counts(dropna = False)

#%%

input_lesion['TIMI pré-PCI'].value_counts(dropna = False)

#%%
input_lesion.columns[0:24]

#%%
#We can have ordinal values
#We can have categorical and target encoding

#%%

input_lesion[['Description de la lésion 1', 'Site de la lésion', '% de sténose']]

#%%
input_lesion['OCT/IVUS réalisée'].value_counts(dropna = False)

#%%
input_lesion[['Description stent restenosé' ,'Stent nu', 'Stent actif', 'Stent biodégradable' ]]


#%%
input_lesion['Description stent restenosé 2'].value_counts(dropna = False)

#%%
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
input_lesion.dtypes

#%%
input_lesion['Longueur total de stent par artère'].iplot(kind='hist', xTitle='Diamètre Lesion', yTitle='count', title='Diamètre lésion')



#%%
input_lesion.columns[0:24]

#%%
Lesion1  = input_lesion.iloc[: , 0:24]
Lesion1.dtypes

#%%
#Ordinal Types
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
Lesion1.iloc[: , 0].value_counts(dropna = False)

#%%
Lesion1.iloc[: , 2].value_counts(dropna = False)

#%%
enc.fit(Lesion1.iloc[: , 0])
enc.categories_

#%%
Occulsion_Enc = [['<50%', 1], ['50-70%', 2], ['70-90%', 3] , ['90-99%', 4],['Occlusion', 5] ]

#%%
Lesion_fill = Lesion1.fillna('missing')
ordinalencoder_X = OrdinalEncoder()
Lesion_fill[:, 0:3] = ordinalencoder_X.fit_transform(Lesion_fill)

#%%
Lesion_fill.dtypes

#%%
Lesion_fill.isna().sum()

#%%
Lesion1.isna().sum()

#%%
Lesion1_numeric = Lesion1.select_dtypes(include=numerics)
#%%
Lesion1_numeric.describe()

#%%
Lesion1.dtypes

#%%
Lesion1['Rotablator'].value_counts(dropna = False)

#%%
Lesion1['Thromboaspiration'].value_counts(dropna = False)

#%%

Lesion1['Ancienneté occlusion'].value_counts(dropna = False)

#%%
Lesion1['Site précis'].value_counts(dropna = False)


#%%
Lesion1['% de sténose'].value_counts(dropna = False)

#%%
#%% [markdown]
#### We have four categories of variables inside each of the 6 Lesion touchées :

## 1- Ordinal Variables : 
## 2- Binary Variables : 
## 3- Numerical Variables : 
## 4- Label Variables :



#%%
# 1 - Ordinal 

#Missing Values

Lesion1['% de sténose'] = Lesion1['% de sténose'].fillna('No Stenose')
#Stenose_categories =['No Stenose', '<50%', '50-70%', '70-90%', '90-99%' , 'Occlusion' ]
#encoder = OrdinalEncoder(categories= Stenose_categories)
encoder = OrdinalEncoder()
Stenose_enc = encoder.fit_transform(Lesion1['% de sténose'].values.reshape(-1, 1))


#%%

Stenose_categories =['No Stenose', '<50%', '50-70%', '70-90%', '90-99%' , 'Occlusion' ]
encoder = OrdinalEncoder(categories= Stenose_categories)
cat = pd.Categorical(Lesion1['% de sténose'], categories=Stenose_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
Stenose_enc = labels


#%%
Lesion1['TIMI post PCI'].value_counts(dropna = False)

#%%
Lesion1['TIMI pré-PCI'] = Lesion1['TIMI pré-PCI'].fillna('No TIMI')
TIMI_categories =['No TIMI' , 'TIMI non précision' , 'TIMI 0', 'TIMI 1',  'TIMI 2', 'TIMI 3'  ]
encoder = OrdinalEncoder(categories= TIMI_categories)
cat = pd.Categorical(Lesion1['TIMI pré-PCI'], categories=TIMI_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
TIMI_pre_enc = labels

#%%
TIMI_pre_enc



#%%
Lesion1['TIMI post PCI'] = Lesion1['TIMI post PCI'].fillna('No reflow')
TIMI_categories =['No reflow' , 'TIMI 0', 'TIMI 1',  'TIMI 2', 'TIMI 3'  ]
encoder = OrdinalEncoder(categories= TIMI_categories)
cat = pd.Categorical(Lesion1['TIMI post PCI'], categories=TIMI_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
TIMI_post_enc = labels



#%%

Lesion1['Résultat angiographique.1'].value_counts(dropna = False)

#%%
Lesion1['Résultat angiographique.1'] = Lesion1['Résultat angiographique.1'].replace('?' , 'No Test')
Lesion1['Résultat angiographique.1'] = Lesion1['Résultat angiographique.1'].fillna('No Test')
Angio_categories =['No Test' , 'Echec', 'Intermédiaire',  'Succès']
encoder = OrdinalEncoder(categories= Angio_categories)
cat = pd.Categorical(Lesion1['Résultat angiographique.1'], categories= Angio_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
Resultat_Angio_enc = labels

#%%
Lesion1['Diamètre lésion'].value_counts(dropna = False)

#%%
Lesion1['Diamètre lésion'] = Lesion1['Diamètre lésion'].fillna('0 mm')
Diametre_categories =['0 mm' , '< 20 mm', '< 25 mm',  '25 mm' , '30 mm' , '35 mm' , '40 mm' , '50 mm']
encoder = OrdinalEncoder(categories= Diametre_categories)
cat = pd.Categorical(Lesion1['Diamètre lésion'], categories= Diametre_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
Diametre_Lesion_enc = labels

#%%
Lesion1['Longueur lésion'].value_counts(dropna = False)

#%%
Lesion1['Longueur lésion'] = Lesion1['Longueur lésion'].fillna('0 mm')
Longueur_categories =['0 mm' , '< 10 mm' , '10-20 mm', '20 mm' ]
encoder = OrdinalEncoder(categories= Longueur_categories)
cat = pd.Categorical(Lesion1['Longueur lésion'], categories= Longueur_categories, ordered=True)
labels, unique = pd.factorize(cat, sort=True)
Longueur_Lesion_enc = labels




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

Label_encoder_columns = ['Site de la lésion','Type de lésion' , 'Site précis' , 'Occlusion chronique' , 'OCT/IVUS réalisée'  , 'Stent nu' , 'Stent actif' , 'Ballon seul' , 'Ballon actif' , 'Rotablator', 'Thromboaspiration',
       'Ancienneté occlusion']

Lesion_Label_Encoder = MultiColumnLabelEncoder(columns = Label_encoder_columns).fit_transform(Lesion_fill)

#%%
Lesion_ordinal_df = pd.DataFrame({'Stenose_Enc': Stenose_enc,
     'Diametre_Lesion_enc': Diametre_Lesion_enc,
     'Longueur_Lesion_enc': Longueur_Lesion_enc , 
     'TIMI_POST_enc' : TIMI_post_enc ,
     'TIMI_PRE_enc' : TIMI_pre_enc ,
      'Resultat_Angio_enc' :Resultat_Angio_enc
    })


#%%

Lesion_Total = pd.concat([Lesion_ordinal_df, Lesion_Label_Encoder], axis=1)



#%%
Lesion_Total.columns
Lesion_Total.head(5)


#%%
Lesion_train = Lesion_Total.drop(['Description de la lésion 1', '% de sténose','Description stent restenosé', 'TIMI pré-PCI' , 'Résultat angiographique.1' , 'TIMI post PCI' , 'Diamètre lésion',
       'Longueur lésion' , 'FFR réalisé' , 'Stent biodégradable'] , axis = 1)


#%%






#%% [markdown]

## INPUT DATA ON PATIENT
#
# Creation of features concerning patients
#
# Label encoding and filing NA

input_train = coro.iloc[ : , 0:50]


#%%
input_train.columns

#%%
input_train.isna().sum()
#%%
features = ['Âge', 'Assistance circulatoire', 'Abord artériel principal',
       'Taille du désilet', 'Fermeture artérielle',
       'Ventriculographie gauche réalisée durant la procédure',
       'FEVG ventriculographie', 'Résultat angiographique',
       'Tronc commun > 50%', 'Pontage > 50%', 'Nb d\'artère(s) dilatée(s)',
       'Nb de site(s) dilaté(s)', 'Nb total de stent(s) implanté(s)',
       'Traitement proposé au patient au décours de la procédure',
       'Syntax_score_global', 'Sexe du patient', 'Taille', 'Poids', 'IMC',
       'ATCD ATC', 'ATCD d\'IDM', 'ATCD d\'AVC',
       'ATCD pathologie vasculaire périphérique', 'Insuffisance rénale',
       'Pontage aorto-coronaire', 'Diabète sucré', 'Dyslipidémie', 'Tabagisme',
       'HTA', 'Hérédité coronaire', 'Ischémie documentée', 'FEVG','ST+<24h']



#%%

input_train.dtypes

#%%

def myfillnaPatient(series):
    if series.dtype is pd.np.dtype(float):
        return series.fillna(series.mean())
    elif series.dtype is pd.np.dtype(object):
        return series.fillna('missing')
    else:
        return series



#%%
input_train_fill = input_train[features].apply(myfillnaPatient)

#%%
input_train_fill.isna().sum()

#%%

input_train_fill.dtypes

#%%

Input_train_cat_columns = input_train_fill.select_dtypes(include=['object']).columns


#%%
input_train_model = MultiColumnLabelEncoder(columns = Input_train_cat_columns).fit_transform(input_train_fill)

#%%

#%% [markdown]

## OUTPUT 
#
# Creation by combing Death , Survival and Disease
#
# The last category of Nan will be used as a new category


#%%

output = coro.iloc[ : , 259:278]
#%%
output.describe(include='all').T

#%%
output['Survie H + 1 an'].value_counts(dropna = False)


#%%
output['Infarctus du myocarde 1an'].value_counts(dropna = False)

#%%
output['Hémorragie grave 1an'].value_counts(dropna = False)

#%%
output['Accident vasculaire cérébral 1an'].value_counts(dropna = False)

#%%
output['Survie H + 1 an'].value_counts(dropna = False)
#%%
#(output['Accident vasculaire cérébral 1an'].str.contains('AVC')) & (~output['Accident vasculaire cérébral 1an'].str.contains('Pas' , na = False))
#%%
output['Target'] = np.where(output['Survie H + 1 an'] == 'Vivant' , 0 , np.where(output['Survie H + 1 an'] == 'Décès H' , 1 , np.where(((output['Hémorragie grave 1an'].str.contains('BARC')) | ((output['Accident vasculaire cérébral 1an'].str.contains('AVC')) & (~output['Accident vasculaire cérébral 1an'].str.contains('Pas' , na = False)))), 2 , 3 )))
#%%
output['Target'] .value_counts(dropna = False)


#%%











#%%
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)






#%% [markdown]

## MODEL CREATION
#
# Model to be created based on Random Forest , SVC for Binary and Infarctus
#
# Ideas for Deep Learning and RNN to be tested

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing

#%%
y_train = output['Target']

#%%
clf = RandomForestClassifier(max_depth=10, random_state=9  , n_estimators=250 , class_weight= 'balanced')

#%%
clf.fit(Lesion_train , y_train)

#%%
pred_cv_label = cross_val_predict(clf, Lesion_train, np.ravel(y_train),
                             cv=10, n_jobs=-1)

#%%
multiclass_roc_auc_score(y_train , pred_cv_label)



#%%
AVC = (output['Accident vasculaire cérébral 1an'].str.contains('AVC')) & (~output['Accident vasculaire cérébral 1an'].str.contains('Pas' , na = False))


#%%
IDM = (output['Infarctus du myocarde 1an'].str.contains('IDM')) & (~output['Infarctus du myocarde 1an'].str.contains('pas' , na = False))


#%%
Deces =(coro['Survie H + 1 an']=='Décès H')


#%%
Deces_H =(coro['Survie H + 1 an']=='Décès 1 an post H')


#%%
Hemmoragie = output['Hémorragie grave 1an'].str.contains('BARC')


#%%

Vivant =(coro['Survie H + 1 an']=='Vivant')

#%%
output['Hémorragie grave 1an'].value_counts(dropna = False)


#%%
(Vivant & Hemmoragie).sum()


#%%
(Vivant & IDM).sum()

#%%
(Vivant & AVC).sum()



#%%
output['Target2'] = np.where((Deces | Deces_H | AVC | IDM | Hemmoragie ) , 1 , np.where((output['Survie H + 1 an'] == 'Vivant') , 0 , 'Nan'))

#%%
y_train2 = output['Target2']


#%%







#%%
clf2 = RandomForestClassifier(max_depth=10, random_state=9  , n_estimators=250 , class_weight= 'balanced')
clf2.fit(Lesion_train, y_train2)

#%%
pred_cv_label2 = cross_val_predict(clf2, Lesion_train, np.ravel(y_train2),
                             cv=10, n_jobs=-1)

#%%
multiclass_roc_auc_score(y_train2 , pred_cv_label2)


#%%
feature_importances = pd.DataFrame(clf2.feature_importances_,
                                   index = Lesion_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)



#%%
feature_importances

#%%





#%% [markdown]

## MODEL CREATION 2
#
# Model to be created based on Random Forest , SVC for Binary and Infarctus
#
# Ideas for Deep Learning and RNN to be tested




coro_train = pd.concat([Lesion_train, input_train_model], axis=1)

#%%
clf3 = RandomForestClassifier(bootstrap= True,
 max_depth= 100,
 max_features= 3,
 min_samples_leaf =3,
 min_samples_split = 8,
 n_estimators = 100, class_weight= 'balanced')
clf3.fit(coro_train, y_train2)

#%%
pred_cv_label3 = cross_val_predict(clf3, coro_train, np.ravel(y_train2),
                             cv=10, n_jobs=-1)

#%%
multiclass_roc_auc_score(y_train2 , pred_cv_label3)

#%%
feature_importances = pd.DataFrame(clf3.feature_importances_,
                                   index = coro_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


#%%
feature_importances


#%% [markdown]

## MODEL CREATION 2
#
# Model to be created based on Random Forest , SVC for Binary and Infarctus
#
# Ideas for Deep Learning and RNN to be tested

#%% 
y_train2.value_counts(dropna = False)

#%%
y_train3 = y_train2.drop()

#%%
index_nan = y_train2.index[y_train2 == 'Nan'].tolist()

#%%
coro_train1 = coro_train.drop(index = index_nan)

#%%
coro_train1.shape


#%%
y_train_binary = y_train2.drop(index = index_nan)

#%%
y_train_binary_num = pd.to_numeric(y_train_binary)


#%%
clf3 = RandomForestClassifier(bootstrap = True,
 max_depth = 12,
 max_features= 3,
 min_samples_leaf= 3,
 min_samples_split = 8,
 n_estimators= 200 ,class_weight= 'balanced')
clf3.fit(coro_train1, y_train_binary)

#%%
pred_cv_label_binary = cross_val_predict(clf3, coro_train1, np.ravel(y_train_binary),
                            method='predict_proba',  cv=10, n_jobs=-1)

#%%
roc_auc_score(y_train_binary , pred_cv_label_binary[:,1])

#%%  [markdown]

# SVC Model to test 



#%% 
from sklearn.svm import SVC



#%%
svc_model = SVC(gamma='auto' , class_weight = 'balanced' , probability = True)

#%%
svc_model.fit(coro_train1, y_train_binary_num)

#%%

pred_cv_svc = cross_val_predict(svc_model, coro_train1, np.ravel(y_train_binary_num),
                            method='predict_proba',  cv=10, n_jobs=-1)


#%%
roc_auc_score(y_train_binary_num , pred_cv_svc[:,1])

#%%



#%%  [markdown]

# Logistic Regression Model  

from sklearn.linear_model import LogisticRegression 

#%% 

LR = LogisticRegression(random_state=9 , solver = 'lbfgs' , max_iter = 100000).fit(coro_train1, y_train_binary_num)  


#%%
pred_cv_LR = cross_val_predict(LR, coro_train1, np.ravel(y_train_binary_num),
                            method='predict_proba',  cv=10, n_jobs=-1)



#%%

import xgboost as xgb
from xgboost import XGBClassifier

alg = XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                        min_child_weight=3, gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)





#%%
seed = 7
np.random.seed(seed)
from scipy import interp  
from sklearn.metrics import roc_curve,auc
#from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import StratifiedKFold, cross_val_score                       
X = coro_train1.reset_index(drop = True)
y = y_train_binary_num.reset_index(drop = True)
cv = StratifiedKFold(n_splits=5)

#%%
roc_auc_score(y_train_binary_num , pred_cv_LR[:,1])

#%%

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = clf3.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = LR.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%

import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]


#%%
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = alg.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#%% [markdown]
## Hyperparameters tuning

# Grid Search and Randomized Search





#%%
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



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
# Fit the grid search to the data
grid_search.fit(coro_train1, y_train_binary)

#%%
grid_search.best_params_


#%%
feature_importances = pd.DataFrame(clf3.feature_importances_,
                                   index = coro_train1.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)



                



#%%
from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

perm_imp_rfpimp = permutation_importances(clf3, coro_train1, y_train_binary, r2)

#%%

def imp_df(column_names, importances):
    df = pd.DataFrame({'feature': column_names,'feature_importance': importances}).sort_values('feature_importance', ascending = False).reset_index(drop = True)
    return df


#%% [markdown]
### This approach is quite an intuitive one, as we investigate the importance of a feature by comparing a model with all features versus a model with this feature dropped for training.

### I created a function (based on rfpimp's implementation) for this approach below, which shows the underlying logic.

## Pros:

### most accurate feature importance
## Cons:

### potentially high computation cost due to retraining the model for each variant of the dataset (after dropping a single feature column)


#%%

from sklearn.base import clone 

def drop_col_feat_imp(model, X_train, y_train, random_state = 42):
    
    # clone the model to have the exact same specification as the one initially trained
    model_clone = clone(model)
    # set random_state for comparability
    model_clone.random_state = random_state
    # training and scoring the benchmark model
    model_clone.fit(X_train, y_train)
    benchmark_score = model_clone.score(X_train, y_train)
    # list for storing feature importances
    importances = []
    
    # iterating over all columns and storing feature importance (difference between benchmark and new model)
    for col in X_train.columns:
        model_clone = clone(model)
        model_clone.random_state = random_state
        model_clone.fit(X_train.drop(col, axis = 1), y_train)
        drop_col_score = model_clone.score(X_train.drop(col, axis = 1), y_train)
        importances.append(benchmark_score - drop_col_score)
    
    importances_df = imp_df(X_train.columns, importances)
    return importances_df

#%%
drop_col_feat_imp(clf3 , coro_train1 , y_train_binary)

#%%
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

#%%
ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)

#%%
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = ada_model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%
ada_model.feature_importances_ 

#%%
from sklearn.ensemble import GradientBoostingClassifier

#%%
grad_model = GradientBoostingClassifier(n_estimators=100)

#%%
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, y):
    probas_ = grad_model.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#%%

coro[coro.dtypes[(coro.dtypes=="float64")|(coro.dtypes=="int64")]
                        .index.values].hist(figsize=[30,30])

#%%
