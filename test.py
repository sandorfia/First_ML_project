import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/mxmh_survey_results.csv')

numerical_df = df.select_dtypes(include=['float64', 'int64'])

numerical_df = numerical_df.dropna()

X = numerical_df.drop(columns=['Depression'])
y = numerical_df['Depression']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print("Training Random Forest Regressor...")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print("Training finished successfully!")
print(f"Mean Squared Error on Test Set: {mse:.2f}")

print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"- {feature:15}: {importance:.3f}")
