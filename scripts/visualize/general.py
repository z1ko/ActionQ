
import pandas as pd

# Load dataset
df = pd.read_parquet('data/processed/kimore.parquet.gzip')
print(df)
