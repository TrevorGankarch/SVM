# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the CSV data
df = pd.read_csv('cleaned_data.csv')  # Replace with your dataset path

# Step 2: Preprocess the data
# Handle missing values (remove or fill as necessary)
df = df.dropna()  # You can also fill missing values with df.fillna(method='ffill')

# Split the data into features (X) and target variable (y)
X = df.drop('Age', axis=1)  # Replace 'target' with the name of your target column
y = df['Heart Disease Status']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Standardize the data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Define the SVM model
svm = SVC()

# Step 6: Define the hyperparameter grid to tune 
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    'degree': [3, 4, 5]  # Degree for 'poly' kernel, if applicable
}

# Step 7: Set up the GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1, scoring='accuracy')

# Step 8: Perform the hyperparameter tuning
grid_search.fit(X_train_scaled, y_train)

# Step 9: Print the best hyperparameters found
print("Best hyperparameters:", grid_search.best_params_)

# Step 10: Use the best model to make predictions
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test_scaled)

# Step 11: Evaluate the model
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report SVM:")
print(classification_report(y_test, y_pred))
