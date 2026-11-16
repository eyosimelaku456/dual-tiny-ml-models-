import pandas as pd

df = pd.read_csv(r'C:\Users\hp\Desktop\intern2\Dataset-model\Dataset-model\EIS_dataset.csv', nrows=5)
print("=== COLUMNS ===")
print(list(df.columns))
print("\n=== SHAPE (sample) ===")
print(df.shape)
print("\n=== FIRST ROW ===")
print(df.iloc[0])

# Full dataset shape
df_full = pd.read_csv(r'C:\Users\hp\Desktop\intern2\Dataset-model\Dataset-model\EIS_dataset.csv')
print("\n=== FULL SHAPE ===")
print(df_full.shape)
print("\n=== DTYPES ===")
print(df_full.dtypes)
print("\n=== NULL COUNT ===")
print(df_full.isnull().sum().sum())
