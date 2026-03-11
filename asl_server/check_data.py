import pandas as pd
df = pd.read_csv('hand_data.csv')
print(df['label'].value_counts())
