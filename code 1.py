import pandas as pd

# 🔹 Step 1: Load both files
df1 = pd.read_excel(r"D:\sriram\Projects\Fake news detection\data\Raw\15_fake_news_detection.csv.xlsx")
df2 = pd.read_csv(r"D:\sriram\Projects\Fake news detection\data\Raw\fake_news_dataset.csv")

print("File 1 shape:", df1.shape)
print("File 2 shape:", df2.shape)

# 🔹 Step 2: Keep only text + label
df1 = df1[['text', 'label']]
df2 = df2[['text', 'label']]

# 🔹 Step 3: Standardize labels (real=0, fake=1)
df1['label'] = df1['label'].str.lower()
df2['label'] = df2['label'].str.lower()

label_map = {'real': 0, 'fake': 1}
df1['label'] = df1['label'].map(label_map)
df2['label'] = df2['label'].map(label_map)

# 🔹 Step 4: Merge datasets
merged_df = pd.concat([df1, df2], axis=0).reset_index(drop=True)

# 🔹 Step 5: Shuffle rows
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 🔹 Step 6: Save final dataset
output_path = r"D:\sriram\Projects\Fake news detection\data\merged_fake_news_dataset.csv"
merged_df.to_csv(output_path, index=False)

print("✅ Final merged dataset saved at:", output_path)
print("Final shape:", merged_df.shape)
print(merged_df.head())
