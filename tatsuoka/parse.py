import pandas as pd

csv_path = 'data/tatsuoka.csv'
df = pd.read_csv(csv_path)
train_df = df.sample(frac=0.8, random_state=200)
test_df = df.drop(train_df.index)
train_df.to_csv('data/train.csv')
test_df.to_csv('data/test.csv')

