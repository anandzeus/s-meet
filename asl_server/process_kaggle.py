import os
import cv2
import mediapipe as mp
import csv
import kagglehub

# Download latest version of the ASL alphabet dataset
print("Downloading Kaggle ASL Alphabet dataset...")
path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Path to dataset files:", path)

# The dataset structure is typically: asl_alphabet_train/asl_alphabet_train/[A-Z, space, del, nothing]
DATA_DIR = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
if not os.path.exists(DATA_DIR):
    # Depending on how the kaggle dataset unzips, we fallback to just the base train dir
    DATA_DIR = os.path.join(path, "asl_alphabet_train")

OUTPUT_FILE = 'hand_data.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

header = ['label'] + [f'coord_{i}' for i in range(42)]

# Overwrite old dataset
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    labels = sorted(os.listdir(DATA_DIR))
    for label in labels:
        folder_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder_path): continue
        
        # We cap at 500 images per class so training doesn't take hours
        print(f"Processing letter/class: {label}...")
        img_names = os.listdir(folder_path)[:500] 
        
        processed_count = 0
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            
            img = cv2.imread(img_path)
            if img is None: continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y])
                    
                    base_x, base_y = coords[0], coords[1]
                    norm_coords = []
                    for i in range(0, len(coords), 2):
                        norm_coords.append(coords[i] - base_x)
                        norm_coords.append(coords[i+1] - base_y)
                    
                    writer.writerow([label] + norm_coords)
                    processed_count += 1
        
        print(f"  -> Extracted {processed_count} valid hand landmarks for '{label}'.")

print(f"Done! Extracted MediaPipe landmarks saved to {OUTPUT_FILE}")
