import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def prepocessing(df):
    print('Processing data...')

    processed_df = pd.get_dummies(df, columns=['Pclass', 'Embarked', 'SibSp', 'Parch'])
    processed_df['Sex'] = processed_df['Sex'].map({'male' : 1 , 'female' : 0})
    processed_df = processed_df.drop(columns=['Name', 'Ticket'])

    processed_df['Cabin'] = processed_df['Cabin'].fillna('other')
    processed_df['Cabin'] = processed_df['Cabin'].apply(lambda x: x[0] if x != 'other' else x)
    processed_df = pd.get_dummies(processed_df, columns=['Cabin'])

    # bin Fare into bins from 0-100, 100-200, 200-300, 300-400, 400-500, 500-600
    processed_df['Fare'] = pd.cut(processed_df['Fare'], bins=[0,100,200,300,600], labels=['100 below','100-200','200-300','300 above'])
    processed_df = pd.get_dummies(processed_df, columns=['Fare'])

    # bin Age into bins of child, adult, old
    processed_df['Age'] = pd.cut(processed_df['Age'], bins=[0,12,60,100], labels=['child','adult','old'])
    processed_df = pd.get_dummies(processed_df, columns=['Age'])

    # convert all columns zero or one
    processed_df = processed_df.apply(lambda x: x.astype(np.int64))
    print('Processing completed.')
    return processed_df

data = pd.read_csv('dataset/train.csv')

processed_data = prepocessing(data)
print('Data shape: ', processed_data.shape)
processed_data.to_csv('dataset/train_processed.csv', index=False)
print('Processed data saved as train_processed.csv')

print('\n')

data = pd.read_csv('dataset/test.csv')
processed_data = prepocessing(data)
print('Data shape: ', processed_data.shape)
processed_data.to_csv('dataset/test_processed.csv', index=False)
print('Processed data saved as test_processed.csv')

exit()