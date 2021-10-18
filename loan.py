#%% Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import scipy as sp
path =  '/home/isaac/Documents/Bases/'
import statsmodels.api as sm
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.inspection import permutation_importance

#%% Load data from the csv file
df = pd.read_csv(path + 'loan_data.csv', index_col=None)


#%% Change the dots in the column names to underscores
df.columns = [c.replace(".", "_") for c in df.columns]
print(f"Number of rows/records: {df.shape[0]}")
print(f"Number of columns/variables: {df.shape[1]}")
df.head()

# Understand your variables

#variables = pd.DataFrame(columns=['Variable', 'Number of unique values', 'Values'])
print(df.info())

#for i, var in enumerate(df.columns):
#    variables.loc[i] = [var, df[var].nunique(), df[var].unique().tolist()]

# Join with the variables dataframe
#var_dict = pd.read_csv('variable_explanation.csv', index_col=0)
#variables.set_index('Variable').join(var_dict)
#%%Loans paid and not_fully_paid
paid = df[df['not_fully_paid'] == 0]
not_paid = df[df['not_fully_paid']==1]
#%% Start coding
#First insights and visualization from the data
print(df.describe())
print(df.value_counts())
purposes = df.groupby('purpose', as_index=False).count()
print(purposes)
numerical = df.select_dtypes(['int64','float64']).columns
numerical = numerical.drop(['credit_policy','not_fully_paid'])
df[numerical].hist(figsize=(20,15), edgecolor= 'white', color = 'green')
plt.show()
#By this, the most of the histograms follows a normal distribution except for some revol_bal, delingq_2yrs,inq_lasth_6mtnhs and pub_rec
#Fix the revol_bal doing logarithmic transformation
np.log(df['revol_bal']+1).hist()
plt.show()
df['revol_bal'] = np.log(df['revol_bal']+1)
df[numerical].hist(figsize=(20,15), edgecolor= 'white', color = 'blue')
plt.show()
#%% Early visualization per paid or not paid
df[df["not_fully_paid"] == 0].drop(['credit_policy',"not_fully_paid"],axis=1).hist(edgecolor= 'white', color = 'blue', alpha = 0.5,figsize=(20,15))
plt.suptitle("Fully paid", size = 30)
plt.show()
df[df["not_fully_paid"] == 1].drop(['credit_policy',"not_fully_paid"],axis=1).hist(edgecolor= 'white', color = 'red', alpha = 0.5, figsize=(20,15))
plt.suptitle("Not fully paid",size=30)
plt.show()
print(paid.describe())
print(not_paid.describe())
#%%Some useful plots
fig = plt.figure(figsize=(10,6))
plot1 = sns.countplot(x = 'purpose',data = df)
plt.xlabel("")
plt.setp(plot1.get_xticklabels(), rotation=45)
plt.title('Number of loans per purpose')
fig.subplots_adjust(bottom=0.3)
plt.show()
plt.close()
###################
fig = plt.figure(figsize=(10,10))
plot2 = sns.countplot(x = 'purpose',hue = 'not_fully_paid',data = df)
plt.setp(plot2.get_xticklabels(), rotation=45)
plt.title(y = 1, label = 'Loans paid or not paid per purpose')
plt.xlabel("")
plt.legend(['Not fully paid', "Fully paid"])
fig.subplots_adjust(bottom=0.2)
plt.show()
plt.close()
##############
fig = plt.figure(figsize=(10,8))
df[df['credit_policy']==0]['fico'].hist(bins=40,alpha = 0.5, label = '0')
df[df['credit_policy']==1]['fico'].hist(bins=40,alpha = 0.5, label = '1')
plt.legend(["Doesn't meet the criteria", "Meets the criteria"])
plt.xlabel('FICO')
plt.ylabel('Count')
plt.title('Distribution about the criterias of LendingClub vs FICO')
plt.show()
plt.close()
###################
fig = plt.figure(figsize=(10,8))
df[df['not_fully_paid']==0]['fico'].hist(bins=40,alpha = 0.5, label = '0')
df[df['not_fully_paid']==1]['fico'].hist(bins=40,alpha = 0.5, label = '1')
plt.legend(["Not fully paid", "Fully paid"])
plt.xlabel('FICO')
plt.ylabel('Count')
plt.title('Distribution about fully paids loans vs FICO')
plt.show()
plt.close()
#%% Let's figure out how much are the variables correlated
def tidy_corr_matrix(corr_mat):
    '''
    :param: Correlation matrix to be converted
    :return: Correlation matrix in tidy format
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['var1', 'var2', 'r']
    corr_mat = corr_mat.loc[corr_mat['var1'] != corr_mat['var2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    return (corr_mat)

corr_matrix = df.select_dtypes(include=['float64', 'int']).corr(method='pearson')
tidy_corr_matrix(corr_matrix).head(10)
#%%How much does the FICO inlfuences the interest rate? in a joinplot
g = sns.jointplot(x = 'fico', y="int_rate", data = df, height=10, kind = 'hex')
plt.title("How much does the FICO influences the interest rate")
g.plot_joint(sns.kdeplot, color = 'r')
plt.show(dpi= 2000)
#%%How much does the interest rate influences the loan get paid or not with a boxplot
fig = plt.figure(figsize=(10,6))
sns.boxplot(x = 'not_fully_paid', y = 'int_rate', data = df)
plt.title("How much does the FICO influences the interest rate")
plt.show(dpi=2000)
#%% Plot a heatmap customized
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
sns.heatmap(
    corr_matrix,
    annot = True,
    cbar = True,
    annot_kws = {"size": 6},
    vmin = -1,
    vmax  = 1,
    center = 0,
    cmap = sns.diverging_palette(20, 220, n=200),
    square = True,
    ax = ax
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
)
ax.tick_params(labelsize = 8)
fig.subplots_adjust(bottom=0.2)
plt.title('Pearson correlation')
plt.show(dpi=2000)
#%%Generate the dataset with dummy variables for the purpose column because its categorical in string type
df1 = pd.get_dummies(df, columns=['purpose'],drop_first=False)
df1.info()
#%% Divide the dataset into train and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df1.drop(columns='not_fully_paid', axis = 'columns'),
                                                    df1['not_fully_paid'],train_size=0.7, random_state=2020,
                                                    shuffle=True)
#%% Comparing the distributions between de tests and train data for the variable who wanna predict
print(y_test.describe())
print(y_train.describe())
#%% Random forest model
rf = RandomForestClassifier(oob_score=True,n_jobs=-1, random_state=2021,n_estimators=600)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
#%%
mat_confusion = confusion_matrix(y_true = y_test, y_pred = pred)
accuracy = accuracy_score(y_true  = y_test, y_pred = pred, normalize = True)
print("Confussion matrix")
print("-------------------")
print(mat_confusion)
print("")
print(f"The accuracy test its: {100 * accuracy} %")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
plot_confusion_matrix(rf,X_test,y_test)
plt.show()

#%% Some metrics about the model
print(classification_report(y_test, pred))
