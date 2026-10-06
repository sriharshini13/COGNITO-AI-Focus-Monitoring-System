import pandas as pd

# Load labels
train_df = pd.read_csv('dataset/Labels/TrainLabels.csv')

# Fix column name spacing issue
train_df.columns = train_df.columns.str.strip()

print("=== First 5 rows ===")
print(train_df.head())

print("\n=== Column Names ===")
print(train_df.columns.tolist())

print("\n=== Engagement Score Distribution ===")
print(train_df['Engagement'].value_counts())

print("\n=== Dataset Size ===")
print(f"Total video clips: {len(train_df)}")

# Map to focus levels
def map_focus(row):
    if row['Engagement'] >= 2:
        return 'High'
    elif row['Boredom'] >= 2 or row['Frustration'] >= 2:
        return 'Low'
    else:
        return 'Medium'

train_df['focus_level'] = train_df.apply(map_focus, axis=1)

print("\n=== Focus Level Distribution ===")
print(train_df['focus_level'].value_counts())

print("\n=== Sample with Focus Labels ===")
print(train_df[['ClipID', 'Engagement', 'Boredom', 'focus_level']].head(10))