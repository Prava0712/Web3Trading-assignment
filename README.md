# Web3Trading-assignment
Objective is to explore the relationship between trader performance and market  sentiment, uncover hidden patterns, and deliver insights that can drive smarter trading  strategies.
 1. Bitcoin Market Sentiment Dataset o Columns: Date, Classification (Fear/Greed
2. Historical Trader Data from Hyperliquid o Columns include: account, symbol, execution price, size, side, time, 
start position, event, closedPnL, leverage, etc.

!pip install --quiet plotly scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
sentiment_path = '/content/Feargreedindex.csv'
trades_path = '/content/historicaldata.csv' 

sent = pd.read_csv(sentiment_path)
trades = pd.read_csv(trades_path)

sent['date'] = pd.to_datetime(sent['date'], errors='coerce')
trades['Timestamp'] = pd.to_datetime(trades['Timestamp'], errors='coerce')

# Quick heads-up: ensure column names exactly match; use trades.columns to inspect
print("Columns in sentiment data:")
print(sent.columns)
print("\nColumns in trades data:")
print(trades.columns)
print(sent.head())
print(trades.head())
# Standardize sentiment labels
sent['classification'] = sent['classification'].str.strip().str.title()
# Convert trades timestamp to date and then to datetime objects for merging
trades['date'] = pd.to_datetime(trades['Timestamp'].dt.date)

# Merge daily sentiment to trades by date
trades = trades.merge(sent[['date','classification']], on='date', how='left')
# Example features
trades['notional'] = trades['Execution Price'] * trades['Size USD']
trades['profit_flag'] = (trades['Closed PnL'] > 0).astype(int)
# Convert trades timestamp to date and then to datetime objects for merging
trades['date'] = pd.to_datetime(trades['Timestamp'].dt.date)

# Daily market-level summary
daily = trades.groupby('date').agg(
total_trades=('Account','count'),
total_notional=('notional','sum'),
profit_rate=('profit_flag','mean')
).reset_index()

# Attach sentiment
daily = daily.merge(sent[['date','classification']], on='date', how='left')

# Per-account summary
acct = trades.groupby('Account').agg(
trades_count=('Account','count'),
win_rate=('profit_flag','mean'),
avg_notional=('notional','mean'),
pnl_sum=('Closed PnL','sum')
).reset_index()
import os

# Create the output directory if it doesn't exist
output_dir = '/content/outputs'
os.makedirs(output_dir, exist_ok=True)

# Profit rate by sentiment
plt.figure(figsize=(6,4))
sns.barplot(x='classification', y='profit_rate', data=daily)
plt.title('Daily Profit Rate by Market Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'profit_rate_by_sentiment.png'))
plt.show()
# Compare mean win_rate when sentiment == Greed vs Fear
grp = daily.groupby('classification')['profit_rate'].agg(['mean','count','std']).reset_index()
print(grp)

# Example: logistic regression (binary profitable trade) using sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

# Create the features before performing logistic regression
trades['notional'] = trades['Execution Price'] * trades['Size USD']
trades['profit_flag'] = (trades['Closed PnL'] > 0).astype(int)

features = ['notional']
# Removed 'leverage' as it does not exist
X = trades[features].fillna(0)
y = trades['profit_flag']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
pred = model.predict_proba(X_test)[:,1]
print('AUC:', roc_auc_score(y_test,pred))

Path('/content/csv_files').mkdir(parents=True, exist_ok=True)
Path('/content/outputs').mkdir(parents=True, exist_ok=True)
daily.to_csv('/content/csv_files/daily_summary.csv', index=False)
acct.to_csv('/content/csv_files/account_summary.csv', index=False)
