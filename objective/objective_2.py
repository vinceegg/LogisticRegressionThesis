# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE  # Recursive Feature Elimination for feature selection
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import warnings

# Load the dataset
file_path = 'datasets/phishing_email.csv'  # Ensure this file is in the same directory
data = pd.read_csv(file_path)  # Load the phishing email dataset

# Step 1: Preprocessing
# Vectorizing the text data using TfidfVectorizer with increased number of features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Increasing max_features to 5000 for richer representation
X = tfidf_vectorizer.fit_transform(data['text_combined'])  # Convert the text data to TF-IDF features

# Labels (target variable)
y = data['label']

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create logistic regression model and simulate improving with increasing data size using RFE
data_sizes = [1000, 5000, 10000, 20000, 40000, 60000, 80000]  # Different dataset sizes to simulate scaling
times = []  # To store time taken for each model training
accuracies = []  # To store accuracy for each dataset size

# Suppress convergence warnings and store results
with warnings.catch_warnings(record=True) as w:
    for size in data_sizes:
        # Reduce the training set size for each iteration to simulate scaling with dataset size
        X_train_subset = X_train[:size]
        y_train_subset = y_train[:size]

        # Initialize Logistic Regression model with hyperparameter tuning
        base_model = LogisticRegression(max_iter=50, solver='lbfgs', random_state=42, class_weight='balanced')  
        # Create logistic regression model with limited iterations and balanced class weights

        # Perform Recursive Feature Elimination (RFE) with a larger step size for faster processing
        rfe = RFE(estimator=base_model, n_features_to_select=500, step=100)  
        # Select top 500 features using RFE and remove 100 features at each step
        start_time = time.time()  # Start timer

        try:
            # Fit RFE with the training subset
            X_train_rfe = rfe.fit_transform(X_train_subset, y_train_subset)  # Apply RFE to the training subset
            X_test_rfe = rfe.transform(X_test)  # Apply the same transformation to the test set

            # Hyperparameter tuning using GridSearchCV to find the best model parameters
            param_grid = {'C': [0.1, 1, 10, 100, 1000]}  # Regularization parameter tuning
            grid_search = GridSearchCV(LogisticRegression(max_iter=100, solver='lbfgs', class_weight='balanced'),
                                       param_grid, cv=3, scoring='accuracy')  # Perform grid search with cross-validation
            grid_search.fit(X_train_rfe, y_train_subset)  # Fit grid search model

            best_model = grid_search.best_estimator_  # Get the best model from the grid search

            # Stop timer and record time
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)  # Store training time

            # Predict on the test set and calculate accuracy
            y_pred = best_model.predict(X_test_rfe)  # Make predictions on the test set
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of the model
            accuracies.append(accuracy)  # Store accuracy

            # Print results for current dataset size
            print(f"Dataset Size: {size}")
            print(f"Training Time: {elapsed_time:.4f} seconds")
            print(f"Accuracy: {accuracy:.4f}")
            print("----------------------------------------")

        except Exception as e:
            # In case the model fails, record NaN for time and accuracy
            times.append(float('nan'))  # Store NaN for training time
            accuracies.append(float('nan'))  # Store NaN for accuracy
            print(f"Failed to train for size {size}: {e}")

    # Print warnings, if any
    for warning in w:
        print(f"Warning: {warning.message}")  # Print warnings if any occurred during model training

# Step 3: Visualize the training times for different dataset sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, times, marker='o', color='b')  # Plot training time vs dataset size
plt.title('Logistic Regression Training Time vs Dataset Size (with RFE and TF-IDF)')  # Set the title of the plot
plt.xlabel('Dataset Size')  # Label for the x-axis
plt.ylabel('Training Time (seconds)')  # Label for the y-axis
plt.grid(True)  # Enable grid on the plot
plt.show()  # Display the plot

# Step 4: Visualize the accuracies for different dataset sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, accuracies, marker='o', color='r')  # Plot accuracy vs dataset size
plt.title('Logistic Regression Accuracy vs Dataset Size (with RFE and TF-IDF)')  # Set the title of the plot
plt.xlabel('Dataset Size')  # Label for the x-axis
plt.ylabel('Accuracy')  # Label for the y-axis
plt.grid(True)  # Enable grid on the plot
plt.show()  # Display the plot
