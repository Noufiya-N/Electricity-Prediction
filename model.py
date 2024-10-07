import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv"
data = pd.read_csv(url, low_memory=False)

# Display column names to ensure we have the correct columns
print("Column Names:", data.columns.tolist())

# Select relevant features for prediction (using the correct case for column names)
X = data[['SystemLoadEA', 'ForecastWindProduction', 'ORKTemperature']]

# Assuming 'SMPEA' is the target variable (electricity price)
y = data['SMPEA']

# Replace non-numeric values ('?') with NaN
X = X.replace('?', pd.NA)
y = y.replace('?', pd.NA)

# Drop rows with missing values
X = X.dropna()
y = y[X.index]

# Convert to appropriate data types
X = X.apply(pd.to_numeric)
y = pd.to_numeric(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('electricity_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training completed and saved.")
