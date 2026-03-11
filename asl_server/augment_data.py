import pandas as pd
import numpy as np
import random

def rotate_landmarks(coords, angle_degrees):
    angle_rad = np.radians(angle_degrees)
    cos_val = np.cos(angle_rad)
    sin_val = np.sin(angle_rad)
    
    rotated = []
    # Landmarks are relative to first point (0,0)
    for i in range(0, len(coords), 2):
        x, y = coords[i], coords[i+1]
        nx = x * cos_val - y * sin_val
        ny = x * sin_val + y * cos_val
        rotated.extend([nx, ny])
    return rotated

def scale_landmarks(coords, scale_factor):
    return [c * scale_factor for c in coords]

def add_noise(coords, noise_level=0.005):
    return [c + random.uniform(-noise_level, noise_level) for c in coords]

INPUT_FILE = 'hand_data.csv'
df = pd.read_csv(INPUT_FILE)

# Labels to augment
target_labels = ['i_love_you', 'hello', 'yes', 'no', 'thankyou', 'please', 'sorry', 'help', 'family', 'house', 'sign language']
target_count = 300

new_rows = []

for label in target_labels:
    samples = df[df['label'] == label]
    if len(samples) == 0:
        continue
    
    current_count = len(samples)
    needed = target_count - current_count
    
    if needed <= 0:
        continue
        
    print(f"Augmenting '{label}': {current_count} -> {target_count}")
    
    for _ in range(needed):
        # Pick a random existing sample
        base_sample = samples.iloc[random.randint(0, current_count - 1)]
        coords = base_sample.iloc[1:].tolist()
        
        # Apply random transforms
        angle = random.uniform(-15, 15)
        scale = random.uniform(0.9, 1.1)
        
        aug_coords = rotate_landmarks(coords, angle)
        aug_coords = scale_landmarks(aug_coords, scale)
        aug_coords = add_noise(aug_coords)
        
        new_rows.append([label] + aug_coords)

augment_df = pd.DataFrame(new_rows, columns=df.columns)
final_df = pd.concat([df, augment_df], ignore_index=True)

final_df.to_csv(INPUT_FILE, index=False)
print(f"Dataset successfully augmented! Total rows: {len(final_df)}")
