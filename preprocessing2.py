import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

data = [train_df, test_df]
passenger_id = [train_df['PassengerId'].to_list(), test_df['PassengerId'].to_list()]

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Cabin'].fillna('Missing', inplace=True)
    dataset['Cabin'] = dataset['Cabin'].str[0]
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Sex'] = dataset['Sex'].map({'male': 0,'female': 1})
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset['Fare'] = dataset['Fare'].apply(lambda x: round(x/10)*10)
    bins = [-0.001, 40, 120, 600]
    labels = [0, 1, 2]
    dataset['Fare'] = pd.cut(dataset['Fare'], bins=bins, labels=labels)

age_df = train_df[['Cabin', 'Age']].dropna().astype({'Age': 'int32'}).groupby(['Cabin'], as_index=False).median().sort_values(by='Age', ascending=False).reset_index(drop=True)
age_df['Age'] = age_df['Age'].apply(lambda x: round(x))
age_df = age_df.set_index('Cabin').to_dict()['Age']

for dataset in data:
    dataset['Age'] = dataset.apply(lambda x: age_df[x['Cabin']] if np.isnan(x['Age']) else x['Age'], axis=1)
    dataset['Age'] = dataset['Age'].astype('int32')


for dataset in data:
    dataset['Age_0'] = 0
    dataset['Age_1'] = 0
    dataset['Age_2'] = 0
    dataset['Age_3'] = 0
    dataset['Age_4'] = 0

    dataset.loc[dataset['Age'] <= 16, 'Age_0'] = 1
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age_1'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age_2'] = 1
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age_3'] = 1
    dataset.loc[dataset['Age'] > 64, 'Age_4'] = 1

prob = train_df.groupby(['Cabin'])['Survived'].mean()
prob_df = pd.DataFrame(prob)
prob_df['Died'] = 1 - prob_df['Survived']
prob_df['Ratio'] = prob_df['Survived'] / prob_df['Died']
prob_df.sort_values(by='Ratio', ascending=False)

scaler = StandardScaler()
prob_df['Ratio'] = scaler.fit_transform(prob_df[['Ratio']])
prob_df.sort_values(by='Ratio', ascending=False)

embarked = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False).reset_index(drop=True)
embarked_df = pd.DataFrame(embarked).set_index('Embarked')
embarked_df['Died'] = 1 - embarked_df['Survived']
embarked_df['Ratio'] = embarked_df['Survived'] / embarked_df['Died']
embarked_df['Ratio'] = scaler.fit_transform(embarked_df[['Ratio']])

pclass = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).reset_index(drop=True)
pclass_df = pd.DataFrame(pclass).set_index('Pclass')

rare = train_df['Title'].value_counts().loc[lambda x : x < 10].index.to_list()
train_df['Title'] = train_df['Title'].replace(rare, 'Rare')
title = train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False).reset_index(drop=True)
title['Death'] = 1 - title['Survived']
title['Ratio'] = title['Survived'] / title['Death']
title = pd.DataFrame(title).set_index('Title')

train_df[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df.loc[train_df['FamilySize'] > 6, 'FamilySize'] = 6
fsize = train_df[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False).set_index('FamilySize')

fare = train_df[['Fare','Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False)
fare_df = pd.DataFrame(fare).set_index('Fare')
fare_df['Died'] = 1 - fare_df['Survived']
fare_df['Ratio'] = fare_df['Survived'] / fare_df['Died']
fare_df['Ratio'] = scaler.fit_transform(fare_df[['Ratio']])

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].map(prob_df['Ratio'].to_dict())
    dataset['Embarked'] = dataset['Embarked'].map(embarked_df['Ratio'].to_dict())
    dataset['Pclass'] = dataset['Pclass'].map(pclass_df['Survived'].to_dict())
    dataset['Title'] = dataset['Title'].map(title['Ratio'].to_dict())
    dataset['FamilySize'] = dataset['FamilySize'].map(fsize['Survived'].to_dict())
    dataset['Fare'] = dataset['Fare'].map(fare_df['Ratio'].to_dict()).astype('float64')

for dataset in data:    
    dataset.drop(['PassengerId', 'Ticket'], axis=1, inplace=True)
    dataset.drop(['Name'], axis=1, inplace=True)
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)
    dataset.drop(['Age'], axis=1, inplace=True)

print('Preprocessing Done!')
train_df.to_csv('dataset/train_preprocessed_2.csv', index=False)
print('Training Data Saved!')
test_df.to_csv('dataset/test_preprocessed_2.csv', index=False)
print('Testing Data Saved!')

exit()