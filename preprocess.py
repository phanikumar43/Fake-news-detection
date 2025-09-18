import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🔹 Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# 🔹 Load dataset
df = pd.read_csv(r"D:\sriram\Projects\Fake news detection\data\merged_fake_news_dataset.csv")

# 🔹 Define preprocessing function
def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # 3. Remove punctuation & numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 4. Tokenize
    words = nltk.word_tokenize(text)
    # 5. Remove stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    # 6. Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

# 🔹 Apply cleaning
df['clean_text'] = df['text'].astype(str).apply(clean_text)

# 🔹 Save cleaned dataset
output_path = r"D:\sriram\Projects\Fake news detection\data\cleaned_fake_news_dataset.csv"
df.to_csv(output_path, index=False)

print("✅ Text preprocessing completed!")
print("Saved cleaned dataset at:", output_path)
print(df[['text','clean_text','label']].head())
