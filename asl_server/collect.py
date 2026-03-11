import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- CONFIGURE THE VOCABULARY ---
# We want A-Z plus 20 common words.
ALPHABET = [chr(i) for i in range(ord('a'), ord('z')+1)]
COMMON_WORDS = [
    "hello", "thank you", "yes", "no", "please", 
    "sorry", "help", "love", "eat", "drink", 
    "water", "more", "done", "good", "bad", 
    "happy", "sad", "tired", "friend", "family"
]
VOCAB = ALPHABET + COMMON_WORDS

DATA_FILE = 'hand_data.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Setup CSV file
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: label, then 21 pairs of x,y coordinates
        header = ['label'] + [f'coord_{i}' for i in range(42)]
        writer.writerow(header)

print("=== ASL Dataset Collector ===")
print("You need to collect at least 50 frames for each class to get 90%+ accuracy.")

for word in VOCAB:
    print(f"\n---> Get ready to sign: '{word}' <---")
    print("Press 'r' to start recording 100 frames for this sign.")
    print("Press 's' to skip this sign.")
    print("Press 'q' to quit.")
    
    # Wait for user to be ready
    skip = False
    quit_prog = False
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Ready for: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'r' to record, 's' to skip", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Data Collector', frame)
        
        key = cv2.waitKey(1)
        if key == ord('r'):
            break
        elif key == ord('s'):
            skip = True
            break
        elif key == ord('q'):
            quit_prog = True
            break
            
    if quit_prog:
        break
    if skip:
        continue
        
    # Recording Loop
    recorded = 0
    target = 100
    while recorded < target:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract coordinates
                coords = []
                for lm in hand_landmarks.landmark:
                    coords.extend([lm.x, lm.y])

                base_x, base_y = coords[0], coords[1]
                norm_coords = []
                for i in range(0, len(coords), 2):
                    norm_coords.append(coords[i] - base_x)
                    norm_coords.append(coords[i+1] - base_y)
                
                with open(DATA_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([word] + norm_coords)
                
                recorded += 1
                
        # Show progress
        cv2.putText(frame, f"Recording: {word} ({recorded}/{target})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Data Collector', frame)
        cv2.waitKey(1)

print("\nData collection finished! Run train.py next.")
cap.release()
cv2.destroyAllWindows()
