import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 1: Load dataset
df = pd.read_csv(r"D:\sriram\Projects\Fake news detection\data\merged_fake_news_dataset.csv")

print("✅ Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())

# 🔹 Step 2: Basic info
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# 🔹 Step 3: Label distribution
print("\nLabel counts:")
print(df['label'].value_counts())

plt.figure(figsize=(6,4))
# Use hue='label' with dodge=False to avoid FutureWarning
sns.countplot(x='label', data=df, hue='label', dodge=False, palette='Set2', legend=False)
plt.xticks([0,1], ['Real (0)', 'Fake (1)'])
plt.title("Distribution of Real vs Fake News")
plt.show()

# 🔹 Step 4: Text length analysis
df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

print("\nText length stats:")
print(df['text_length'].describe())

print("\nWord count stats:")
print(df['word_count'].describe())

plt.figure(figsize=(6,4))
sns.histplot(df['word_count'], bins=50, color='skyblue')
plt.title("Distribution of Word Count in Articles")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# 🔹 Step 5: Sample preview
print("\n🔍 Example Real News:")
print(df[df['label']==0].sample(1).text.values[0])

print("\n🔍 Example Fake News:")
print(df[df['label']==1].sample(1).text.values[0])
