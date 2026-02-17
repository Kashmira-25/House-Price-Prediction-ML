import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("Loading dataset...")
df = pd.read_excel("HousePricePrediction.xlsx")

# Remove ID
df.drop("Id", axis=1, inplace=True)

# Fill missing target values
df["SalePrice"].fillna(df["SalePrice"].mean(), inplace=True)

# Drop rows with nulls
df.dropna(inplace=True)

# Target and features
import numpy as np
y = np.log1p(df["SalePrice"])   # log transform target
X = df.drop("SalePrice", axis=1)
# ⭐ Feature Engineering (VERY IMPORTANT)
X["HouseAge"] = 2024 - df["YearBuilt"]
X["RemodelAge"] = 2024 - df["YearRemodAdd"]
X["TotalArea"] = df["LotArea"] + df["TotalBsmtSF"]

# Separate column types
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# Split data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
     "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression(),
    "SVR": SVR()
}

best_model = None
best_score = 999999

print("Training multiple models...\n")

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    
    pipeline.fit(X_train, y_train)
    preds = np.expm1(pipeline.predict(X_test))
    actual = np.expm1(y_test)
    error = mean_absolute_error(actual, preds)
   
    
    print(f"{name} MAE: {error}")
    
    if error < best_score:
        best_score = error
        best_model = pipeline

print("\nSaving best model...")
pickle.dump(best_model, open("model.pkl","wb"))

print("🎉 Training complete! Best model saved.")