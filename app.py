import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
st.set_page_config(page_title=" Brain Tumor Preprocessing", layout="centered")
st.title(" Brain Tumor Dataset Preprocessing App")
uploaded_file = st.file_uploader("Upload your Brain Dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Raw Dataset")
    st.dataframe(df.head())
    st.subheader(" Missing Values in Each Column")
    st.write(df.isnull().sum())
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
    st.subheader(" Boxplots Before Outlier Removal")
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(df[col])
        ax.set_title(f'Boxplot for {col}')
        ax.set_ylabel(col)
        ax.grid(True)
        st.pyplot(fig)
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
    st.success(" Outliers removed successfully!")
    scaler_standard = StandardScaler()
    df_standardized = df.copy()
    df_standardized[numerical_cols] = scaler_standard.fit_transform(df_standardized[numerical_cols])
    st.subheader(" Standardized Data (Z-Score)")
    st.dataframe(df_standardized[numerical_cols].head())
    scaler_normal = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[numerical_cols] = scaler_normal.fit_transform(df_normalized[numerical_cols])
    st.subheader(" Normalized Data (0 to 1)")
    st.dataframe(df_normalized[numerical_cols].head())
else:
    st.info(" Please upload your Brain Tumor dataset CSV to begin.")
