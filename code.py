import pandas as pd

# Load Excel file
df1 = pd.read_excel(r"D:\sriram\Projects\Fake news detection\data\Raw\15_fake_news_detection.csv.xlsx")

# Load CSV file
df2 = pd.read_csv(r"D:\sriram\Projects\Fake news detection\data\Raw\fake_news_dataset.csv")

print("File 1 shape:", df1.shape)
print("File 2 shape:", df2.shape)
