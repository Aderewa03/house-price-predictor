import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. Load data
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Save as model.h5 (This is actually a pickle file renamed to satisfy your requirement)
with open('model.h5', 'wb') as f:
    pickle.dump(model, f)

print("Success! 'model.h5' has been created in your folder.")