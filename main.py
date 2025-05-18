import cv2
import mediapipe as mp
import pickle
import math
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load known hand data if exists
if os.path.exists("hand_db.pkl"):
    with open("hand_db.pkl", "rb") as f:
        hand_db = pickle.load(f)
else:
    hand_db = {}

# Function to compute distance between two sets of landmarks (Euclidean distance)
def compute_distance(lm1, lm2):
    if len(lm1) != len(lm2):
        return float('inf')
    dist = 0
    for p1, p2 in zip(lm1, lm2):
        dist += math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist / len(lm1)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save hand with your name. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    name_found = "Unknown"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Compare with known hands
            min_dist = float('inf')
            for name, ref_lm_list in hand_db.items():
                dist = compute_distance(lm_list, ref_lm_list)
                if dist < min_dist:
                    min_dist = dist
                    name_found = name if dist < 20 else "Unknown"

            # Draw landmarks and label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, name_found, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            # Save new hand with key 's'
            key = cv2.waitKey(1)
            if key == ord('s'):
                name = input("Enter your name: ")
                hand_db[name] = lm_list
                with open("hand_db.pkl", "wb") as f:
                    pickle.dump(hand_db, f)
                print(f"Saved hand landmarks for {name}.")

    cv2.imshow("Hand Recognition", frame)

    # Exit with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
