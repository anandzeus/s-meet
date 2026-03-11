import requests
import pandas as pd

df = pd.read_csv('hand_data.csv')
word_samples = df[df['label'] == 'i_love_you'].head(5)

for i, row in word_samples.iterrows():
    label = row['label']
    coords = row[1:].tolist()
    landmarks = []
    for j in range(0, len(coords), 2):
        landmarks.append({'x': coords[j], 'y': coords[j+1]})
    
    resp = requests.post('http://localhost:8000/asl/predict_alpha', json={'landmarks': landmarks})
    print(f"Sample {i} (Real: {label}) -> Predicted: {resp.json()}")
