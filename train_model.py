import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv("C:/Users/HP/OneDrive/Desktop/cts/data/brain numerical dataset/Brain_Tumor_Prediction_Dataset.csv")
print(" Missing values in each column:")
print(df.isnull().sum())
numerical_cols = ['Age', 'Tumor_Size', 'Genetic_Risk', 'Survival_Rate(%)']
for col in numerical_cols:
    df[col].fillna(df[col].mean(), inplace=True)
categorical_cols = ['Gender', 'Country', 'Tumor_Location', 'MRI_Findings',
                    'Smoking_History', 'Alcohol_Consumption', 'Radiation_Exposure',
                    'Head_Injury_History', 'Chronic_Illness', 'Diabetes',
                    'Tumor_Type', 'Treatment_Received', 'Tumor_Growth_Rate',
                    'Family_History', 'Symptom_Severity', 'Brain_Tumor_Present']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\n Boxplots before removing outliers:")
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col])
    plt.title(f"Boxplot for {col}")
    plt.ylabel(col)
    plt.grid(True)
    plt.show()
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = data[(data[column] >= lower) & (data[column] <= upper)]
    return filtered
for col in numerical_cols:
    df = remove_outliers_iqr(df, col)
print(" Outliers removed!")
scaler_standard = StandardScaler()
df_standardized = df.copy()
df_standardized[numerical_cols] = scaler_standard.fit_transform(df_standardized[numerical_cols])
print(" Standardization done.")
scaler_normal = MinMaxScaler()
df_normalized = df.copy()
df_normalized[numerical_cols] = scaler_normal.fit_transform(df_normalized[numerical_cols])
print(" Normalization done.")
print("\n Standardized values sample:")
print(df_standardized[numerical_cols].head())
print("\n Normalized values sample:")
print(df_normalized[numerical_cols].head())
