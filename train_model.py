import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
df = pd.read_csv("diamonds.csv")

# 2. Features & target
X = df.drop("price", axis=1)
y = df["price"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Preprocessing
categorical_cols = ["cut", "color", "clarity"]
numerical_cols = ["carat", "depth", "table", "x", "y", "z"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# 5. Model pipeline
model = Pipeline([
    ("preprocessing", preprocessor),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

# 6. Train model
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 8. Save model
with open("diamond_knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")