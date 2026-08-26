import pandas as pd


# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)


# Display the first 5 rows
print("========== FIRST 5 ROWS ==========")
print(df.head())


# Display dataset information
print("\n========== DATA INFO ==========")
print(df.info())


# Display missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# Display dataset shape
print("\n========== DATA SHAPE ==========")
print(df.shape)