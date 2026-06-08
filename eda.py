import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. CREATE MOCK FILE (For demonstration purposes)
def create_mock_dataset(filename):
    np.random.seed(42)
    n_rows = 200
    
    data = {
        'Date': pd.date_range(start='2025-01-01', periods=n_rows, freq='D'),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_rows),
        'Age': np.random.randint(18, 65, size=n_rows),
        'Annual_Income': np.random.normal(60000, 15000, size=n_rows).round(2),
        'Purchase_Amount': np.random.uniform(10, 500, size=n_rows).round(2),
        'Satisfaction_Score': np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan], n_rows, p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"--> Success: Mock dataset created as '{filename}'\n")

# Define file path
csv_file = 'data_set.csv'
create_mock_dataset(csv_file)


# 2. FILE INPUT: LOAD AND CLEAN DATA
print("="*50)
print("1. LOADING & CLEANING DATA")
print("="*50)

# Load file
df = pd.read_csv(csv_file)
df['Date'] = pd.to_datetime(df['Date'])

# Show initial structure to terminal
print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Handle missing data (Impute numeric missing score with its median)
median_score = df['Satisfaction_Score'].median()
df['Satisfaction_Score'] = df['Satisfaction_Score'].fillna(median_score)
print(f"\nHandled missing data. Imputed 'Satisfaction_Score' nulls with median: {median_score}")


# 3. TERMINAL OUTPUT: STATISTICAL SUMMARIES
print("\n" + "="*50)
print("2. STATISTICAL SUMMARIES & INSIGHTS")
print("="*50)

# Summary of numerical data
print("\nDescriptive Statistics:")
print(df[['Age', 'Annual_Income', 'Purchase_Amount', 'Satisfaction_Score']].describe().round(2))

# Pattern analysis by group
print("\nAverage Purchase Amount and Satisfaction by Category:")
category_summary = df.groupby('Category')[['Purchase_Amount', 'Satisfaction_Score']].mean().round(2)
print(category_summary)


# 4. IMAGE OUTPUT: VISUALIZATIONS & CORRELATIONS
print("\n" + "="*50)
print("3. GENERATING VISUALIZATIONS (IMAGES)")
print("="*50)

# Set style for charts
sns.set_theme(style="whitegrid")

# Chart 1: Distribution & Category Trends (Subplots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['Purchase_Amount'], kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Purchase Amounts')
axes[0].set_xlabel('Purchase Amount ($)')

sns.boxplot(x='Category', y='Purchase_Amount', data=df, ax=axes[1], palette='Set2')
axes[1].set_title('Purchase Amount Dispersion across Categories')
axes[1].set_xlabel('Product Category')

plt.tight_layout()
plt.show()

# Chart 2: Correlation Heatmap
plt.figure(figsize=(6, 4))
numeric_cols = df[['Age', 'Annual_Income', 'Purchase_Amount', 'Satisfaction_Score']]
correlation_matrix = numeric_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Customer Attributes')

plt.tight_layout()
plt.show()

print("\n--> EDA Project Execution Complete successfully!")
