# Python Implementation of Improved ML Model with fee_ratio Feature

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load your dataset
df = pd.read_csv('your_dataset.csv')

# Handling Missing Values
# Drop rows completely empty except entity_id
df = df.dropna(subset=['primary_org', '23_aum', '24_aum', '23_revenue', '24_revenue', 'open_nnb', 'open_nnbf'], how='all')

# Impute remaining NaNs with median values
df.fillna(df.median(numeric_only=True), inplace=True)

# Feature Engineering
df['aum_growth'] = (df['24_aum'] - df['23_aum']) / df['23_aum'].replace(0, np.nan)
df['revenue_growth'] = (df['24_revenue'] - df['23_revenue']) / df['23_revenue'].replace(0, np.nan)
df['fee_ratio'] = df['open_nnbf'] / df['open_nnb'].replace(0, np.nan)

# Replace infinite and NaNs resulted from division by zero
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Defining Ranking Metric (Customize based on your business logic)
df['rank_metric'] = df['aum_growth'] + df['revenue_growth'] + df['fee_ratio']

# Ranking entities
df['rank_percentile'] = df['rank_metric'].rank(pct=True)

# Labeling
conditions = [
    (df['rank_percentile'] >= 0.99),
    (df['rank_percentile'] <= 0.80)
]
choices = ['top_1%', 'bottom_80%']
df['rank_label'] = np.select(conditions, choices, default='rest_19%')

# Splitting features and target
features = ['23_aum', '24_aum', '23_revenue', '24_revenue', 'open_nnb', 'open_nnbf',
            'aum_growth', 'revenue_growth', 'fee_ratio']
X = df[features]
y = df['rank_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Random Forest)
model = RandomForestClassifier(random_state=42, n_estimators=150, max_depth=10, min_samples_leaf=5)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
