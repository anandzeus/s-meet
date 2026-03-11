import os
import cv2
import mediapipe as mp
import csv
import kagglehub

# We want to add some common words if available
# Dataset: nikhilgawai/sign-language-dataset (contains hello, yes, no, thank you etc.)
print("Downloading additional Sign Language dataset for words...")
try:
    path = kagglehub.dataset_download("nikhilgawai/sign-language-dataset")
    print("Path to dataset files:", path)
except Exception as e:
    print(f"Could not download words dataset: {e}")
    exit(1)

# Inspect the directory structure
# This dataset usually has folders for 'hello', 'yes', 'no', etc.
# We'll search for folders not in the current alphabet list.

OUTPUT_FILE = 'hand_data.csv'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Append to existing dataset
with open(OUTPUT_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    
    # Iterate through download path to find folders
    for root, dirs, files in os.walk(path):
        for label in dirs:
            # We skip alphabet letters already processed if they are duplicated
            if len(label) == 1 and label.isupper():
                continue
                
            folder_path = os.path.join(root, label)
            print(f"Processing word/gesture: {label}...")
            
            img_names = os.listdir(folder_path)[:200]
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
            print(f"  -> Extracted {processed_count} landmarks for '{label}'.")

print("Finished appending word data!")
