import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data (Using California Housing as the example)
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # Price in $100k

# 2. Preprocessing
# Neural Networks REQUIRE scaling. 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Build Neural Network
model = Sequential()
# Input layer (8 features) -> Hidden Layer 1
model.add(Dense(64, activation='relu', input_shape=(8,)))
# Hidden Layer 2
model.add(Dense(32, activation='relu'))
# Output Layer (1 number: Price)
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train
print("Training Neural Network...")
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# 5. Save the required files
model.save('model.h5')  # <--- The file your professor asked for
joblib.dump(scaler, 'scaler.pkl') # We also need this to scale new inputs later!

print("Success! 'model.h5' and 'scaler.pkl' have been created.")