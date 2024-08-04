import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dataset
data = pd.read_csv('data/employee_data.csv')

# Define features and target
features = ['employee_id', 'hours_worked', 'tasks_completed', 'department']
numeric_features = ['hours_worked', 'tasks_completed']
categorical_features = ['employee_id', 'department']

X = data[features]
y = data['anomaly']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Fit the preprocessor on the training data
preprocessor.fit(X_train)

# Save the preprocessor
joblib.dump(preprocessor, 'models/preprocessor.pkl')

# Preprocess the training data
X_train_processed = preprocessor.transform(X_train)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train_processed)

# Save the model
joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')

# Define Autoencoder
input_dim = X_train_processed.shape[1]
encoding_dim = 14  # Change as necessary

input_layer = layers.Input(shape=(input_dim,))
encoder = layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = models.Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(X_train_processed, X_train_processed, epochs=50, batch_size=32, validation_split=0.2)

# Save the model
autoencoder.save('models/autoencoder_model.h5')
