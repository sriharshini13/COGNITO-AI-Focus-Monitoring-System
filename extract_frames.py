import cv2
import os
import pandas as pd

# Load labels
train_df = pd.read_csv('dataset/Labels/TrainLabels.csv')
train_df.columns = train_df.columns.str.strip()

# Map focus levels
def map_focus(row):
    if row['Engagement'] >= 2:
        return 'High'
    elif row['Boredom'] >= 2 or row['Frustration'] >= 2:
        return 'Low'
    else:
        return 'Medium'

train_df['focus_level'] = train_df.apply(map_focus, axis=1)

# Settings
TRAIN_DIR = 'dataset/DataSet/Train'
OUTPUT_DIR = 'frames'
FRAME_INTERVAL = 10  # extract every 10th frame

os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []
total = 0
skipped = 0

# Loop through users
for user_folder in os.listdir(TRAIN_DIR):
    user_path = os.path.join(TRAIN_DIR, user_folder)
    if not os.path.isdir(user_path):
        continue

    # Loop through clip folders
    for clip_folder in os.listdir(user_path):
        clip_path = os.path.join(user_path, clip_folder)
        video_file = os.path.join(clip_path, clip_folder + '.avi')

        if not os.path.exists(video_file):
            skipped += 1
            continue

        # Get label for this clip
        clip_id = clip_folder + '.avi'
        match = train_df[train_df['ClipID'] == clip_id]
        if match.empty:
            skipped += 1
            continue

        focus_label = match.iloc[0]['focus_level']

        # Open video
        cap = cv2.VideoCapture(video_file)
        frame_count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % FRAME_INTERVAL == 0:
                # Save frame
                frame_name = f"{clip_folder}_frame{frame_count}.jpg"
                frame_path = os.path.join(OUTPUT_DIR, frame_name)
                cv2.imwrite(frame_path, frame)
                records.append({
                    'frame_path': frame_path,
                    'clip_id': clip_id,
                    'focus_level': focus_label
                })
                saved += 1

            frame_count += 1

        cap.release()
        total += 1

        if total % 50 == 0:
            print(f"Processed {total} clips so far...")

print(f"\nDone! Processed {total} clips, skipped {skipped}")
print(f"Total frames extracted: {len(records)}")

# Save records to CSV
records_df = pd.DataFrame(records)
records_df.to_csv('features/frames_index.csv', index=False)
print("Saved frames index to features/frames_index.csv")