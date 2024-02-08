from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd

# Assuming df is your DataFrame with 18 categorical columns and a target column 'target'
# Replace 'target' with your actual target column name

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Apply SelectKBest class to extract top 10 best features
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(X, y)

# Get feature scores
feature_scores = pd.DataFrame(fit.scores_)
feature_columns = pd.DataFrame(X.columns)

# Concatenate the dataframes for better visualization
feature_scores_df = pd.concat([feature_columns, feature_scores], axis=1)
feature_scores_df.columns = ['Feature', 'Score']

# Sort features by score in descending order
feature_scores_df.sort_values(by='Score', ascending=False, inplace=True)

# Display top 10 features
print(feature_scores_df.head(10))
