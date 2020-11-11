# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 13:13:08 2020

@author: Shayantani Kar
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Libraries for data visualization
import matplotlib.pyplot as pplt  
import seaborn as sns 

# Import scikit_learn module for the algorithm/model: Linear Regression
from sklearn.linear_model import LogisticRegression
# Import scikit_learn module to split the dataset into train.test sub-datasets
from sklearn.model_selection import train_test_split 

# Import scikit_learn module for k-fold cross validation

# import the metrics class
from sklearn import metrics

#laod the dataset provided
salary_dataset  = pd.read_csv('adult.csv')

# describe the dataset 
print(salary_dataset.describe())

# salary dataset info to find columns and count of the data 
print(salary_dataset.info())


#We count the number of missing values for each feature
print(salary_dataset.isnull().sum())
#below sum shows there are no null values in the dataset so, no need to clean the dataset 

#creating a Dataframe from the given dataset
df = pd.DataFrame(salary_dataset)
print(df.columns)

#replacing some special character columns names with proper names 
df.rename(columns={'capital-gain': 'capital gain', 'capital-loss': 'capital loss', 'native-country': 'country',
                   'hours-per-week': 'hours per week','marital-status': 'marital'}, inplace=True)
print(df.columns)

#Finding the special characters in the data frame 
print(df.isin(['?']).sum(axis=0))
#we see that there is a special character as " ?" for columns workcalss, Occupation, and country
#we need to clean those data 


#assinging the data set to a train data set to remove special characters
#train_data=[salary_dataset]
print(df.columns)

# the code will replace the special character to nan and then drop the columns 
df['country'] = df['country'].replace('?',np.nan)
df['workclass'] = df['workclass'].replace('?',np.nan)
df['occupation'] = df['occupation'].replace('?',np.nan)

#dropping the nan columns now 
df.dropna(how='any',inplace=True)

#Finding if special characters are present in the data 
print(df.isin(['?']).sum(axis=0))

#running a loop for value_counts of each column to find out unique values. 
for c in df.columns:
    print ("---- %s ---" % c)
    print (df[c].value_counts())
    
#checking the Special characters still exists 
df2=df.workclass.value_counts()
import matplotlib.pyplot as plt

k=pd.DataFrame(df2)
k1=k.index.values.tolist()
#k2=k[:,0].values
plt.plot(df2)
plt.show()
df2.plot.bar()
plt.show()
#checking the Special characters still exists 
print(df.occupation.value_counts())

#dropping un-used data from the dataset 
df.drop(['educational-num','age', 'hours per week', 'fnlwgt', 'capital gain','capital loss', 'country'], axis=1, inplace=True)

# Let's see how many unique categories we have in this property
income = set(df['income'])
print(income)
#mapping the data into numerical data using map function
df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

#check the data is replaced 
print(df.head())
# Let's see how many unique categories we have in this gender property
gender = set(df['gender'])
print(gender)

#Mapping the values to numerical values 
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1}).astype(int)

# How many unique races we got here?
race = set(df['race'])
print(race)

#Mapping the values to numerical values 
df['race'] = df['race'].map({'Black': 0, 'Asian-Pac-Islander': 1, 'Other': 2, 'White': 3, 
                                             'Amer-Indian-Eskimo': 4}).astype(int)

# How many unique races we got here?
Marital = set(df['marital'])
print(Marital)

#Mapping the values to numerical values 
df['marital'] = df['marital'].map({'Married-spouse-absent': 0, 'Widowed': 1, 
                                                             'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4, 
                                                             'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)

# How many unique Workclass we got here?
emp = set(df['workclass'])
print(emp)
#Mapping the values to numerical values
df['workclass'] = df['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1, 
                                                             'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4, 
                                                             'Private': 5, 'Self-emp-not-inc': 6}).astype(int)

# How many unique Education we got here?
ed = set(df['education'])
print(ed)

#Mapping the values to numerical values
df['education'] = df['education'].map({'Some-college': 0, 'Preschool': 1, 
                                                        '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, 
                                                        '12th': 5, '7th-8th': 6, 'Prof-school': 7,
                                                        '1st-4th': 8, 'Assoc-acdm': 9,
                                                        'Doctorate': 10, '11th': 11,
                                                        'Bachelors': 12, '10th': 13,
                                                        'Assoc-voc': 14,
                                                        '9th': 15}).astype(int)
# Let's see how many unique categories we have in this Occupation property after cleaning it 
occupation = set(df['occupation'])
print(occupation)
# Now we classify them as numbers instead of their names.
df['occupation'] = df['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 
                                          'Adm-clerical': 3, 'Handlers-cleaners': 4, 
                                         'Prof-specialty': 5,'Machine-op-inspct': 6, 
                                         'Exec-managerial': 7, 
                                         'Priv-house-serv': 8,
                                         'Craft-repair': 9, 
                                         'Sales': 10, 
                                         'Transport-moving': 11, 
                                         'Armed-Forces': 12, 
                                         'Other-service': 13,  
                                         'Protective-serv': 14}).astype(int)
    
    
# How many unique Relationship we got here?
relationship = set(df['relationship'])
print(relationship)


#Mapping the values to numerical values
df['relationship'] = df['relationship'].map({'Not-in-family': 0, 'Wife': 1, 
                                                             'Other-relative': 2, 
                                                             'Unmarried': 3, 
                                                             'Husband': 4, 
                                                             'Own-child': 5}).astype(int)
print(df.head(10))
#Now below we see all the data is numerical data that is proper for our data feature analysis 

#plotting a bar graph for Education against Income to see the co-relation between these columns 
df.groupby('education').income.mean().plot(kind='bar')
plt.show()
#plotting a bar graph for Occupation against Income to see the co-relation between these columns 
df.groupby('occupation').income.mean().plot(kind='bar')

plt.show()

#plotting a bar graph for Relationship against Income to see the co-relation between these columns 
df.groupby('relationship').income.mean().plot(kind='bar')
plt.show()

#plotting a bar graph for Race against Income to see the co-relation between these columns 
df.groupby('race').income.mean().plot(kind='bar')
plt.show()

#plotting a bar graph for Race against Income to see the co-relation between these columns 
df.groupby('gender').income.mean().plot(kind='bar')
plt.show()

#plotting a bar graph for Race against Income to see the co-relation between these columns 
df.groupby('workclass').income.mean().plot(kind='bar')
plt.show()
#plotting a bar graph for Race against Income to see the co-relation between these columns 
df.groupby('marital').income.mean().plot(kind='bar')
plt.show()


# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
corrmat = df.corr()
f, ax = pplt.subplots(figsize=(12, 9))
k = 8 #number of variables for heatmap
cols = corrmat.nlargest(k, 'income')['income'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
pplt.show()

#below we see that there is relation between Relationship, Education, Race, Occupation and Income which is our target 
#columns to predict so, doing more feature analysis on these columns 

# Plot histogram for each numeric variable/attribute of the dataset

df.hist(figsize=(12,9))
pplt.show()

# Density plots

df.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=True, fontsize=1, figsize=(12,16))
pplt.show()

df.columns

#Transform the data set into a data frame 
#NOTE: cleaned_data = the data we want, 
#      X axis = We concatenate the Relationship, Education,Race,Occupation columns using np.c_ provided by the numpy library
#      Y axis = Our target variable or the income of adult i.e Income
df_x = pd.DataFrame(df)
df_x = pd.DataFrame(np.c_[df['relationship'], df['education'], df['race'],df['occupation'],df['gender'],df['marital'],df['workclass']], 
                    columns = ['relationship','education','race','occupation','gender','marital','workclass'])
df_y = pd.DataFrame(df.income)


#Initialize the linear regression model
reg = LogisticRegression()
#Split the data into 67% training and 33% testing data
#NOTE: We have to split the dependent variables (x) and the target or independent variable (y)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)
#Train our model with the training data
reg.fit(x_train, y_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


#print our price predictions on our test data
y_pred = reg.predict(x_test)

# Store dataframe values into a numpy array
array = df.values

# Separate array into input and output components by slicing
# For X (input) [:, 0:0] = all the rows, columns from 0 - 13
# Independent variables - input
X = array[:, 0:6]

# For Y (output) [:, 7] = all the rows, columns index 7 (last column)
# Dependent variable = output
Y = array[:,7]

#df['relationship'], df['education'], df['race'],df['occupation'],df['gender'],df['marital'],df['workclass']
pred1=reg.predict([[5,11,0,6,0,5,5]])
print("Predicttion 1:",pred1)

#Predicting the target value that is if income is <=50K then 0 if not 1 with x-axis columns as given below
pred2=reg.predict([[1,7,3,7,0,2,0]])
print("Predicttion 2:",pred2)

#Predicting the target value that is if income is <=50K then 0 if not 1 with x-axis columns as given below
pred3=reg.predict([[4,12,3,7,0,0,0]])
print("Predicttion 3:",pred3)

#confusion matrix 
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



      

'''https://www.kaggle.com/marksman/us-adult-income-salary-prediction/execution
https://www.kaggle.com/anirudhraj/adult-salary-predictor/data?select=adult.csv
https://towardsdatascience.com/a-beginners-guide-to-data-analysis-machine-learning-with-python-adult-salary-dataset-e5fc028b6f0a
'''