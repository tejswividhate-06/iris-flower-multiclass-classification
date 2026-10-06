import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Save trained model
pickle.dump(model, open("MCModel.pkl", "wb"))

print("MCModel.pkl saved successfully!")