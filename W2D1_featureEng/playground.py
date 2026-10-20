import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Synthetic student ML exam dataset
np.random.seed(7)
n = 80
df = pd.DataFrame({
    'age':         np.random.randint(19, 28, n),
    'study_hrs':   np.random.uniform(1, 8, n).round(1),
    'assignments': np.random.randint(0, 10, n),
    'midterm':     np.random.randint(30, 100, n),
    'final_score': None,
    'grade':       None,
})
df['final_score'] = (df['study_hrs'] * 8 + df['midterm'] * 0.5 + np.random.randn(n) * 5).clip(0, 100).round(1)
df.loc[np.random.choice(n, 8), 'study_hrs'] = np.nan  # add missing
df['grade'] = pd.cut(df['final_score'], bins=[0,59,74,89,100], labels=['F','C','B','A'])

print('=== STEP 1: Shape & Types ===')
print(df.info())

print('\n=== STEP 2: Missing Values ===')
print((df.isnull().mean()*100).round(1).rename('missing_%'))

print('\n=== STEP 3: Descriptive Stats ===')
print(df.describe().round(2))

print('\n=== STEP 4: Target Distribution ===')
print(df['grade'].value_counts())
print(f'Pass rate: {(df["final_score"] >= 60).mean():.1%}')

print('\n=== STEP 5: Correlation ===')
print(df[['study_hrs','midterm','assignments','final_score']].corr().round(2))
