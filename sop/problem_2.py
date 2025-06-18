# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer  # Using CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import warnings

# Load the dataset
file_path = 'datasets/phishing_email.csv'  # Ensure this file is in the same directory
data = pd.read_csv(file_path)  # Load the phishing email dataset

# Step 1: Preprocessing
# Vectorizing the text data using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)  # Increase features to 5000 to make it harder
X = vectorizer.fit_transform(data['text_combined'])  # Convert text data to numerical format (word counts)

# Labels
y = data['label']  # Extract labels (phishing or not)

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
# Split the data into training and testing sets

# Step 2: Create logistic regression model and simulate struggling with increasing data size
data_sizes = [1000, 5000, 10000, 20000, 40000, 60000, 80000]  # Different dataset sizes to simulate struggle
times = []  # To store time taken for each model training
accuracies = []  # To store accuracy for each dataset size

# Suppress convergence warnings and store results
with warnings.catch_warnings(record=True) as w:
    # Limit the iterations to simulate struggle with convergence
    for size in data_sizes:
        X_train_subset = X_train[:size]  # Use only a subset of the training data
        y_train_subset = y_train[:size]  # Corresponding labels for the subset
        
        # Simulate struggle by using a different solver, lowering iterations, and reducing tolerance
        model = LogisticRegression(max_iter=5, solver='liblinear', tol=1e-5)  # Slower solver, fewer iterations, lower tolerance
        start_time = time.time()  # Start timer to track training time
        
        try:
            # Train model
            model.fit(X_train_subset, y_train_subset)  # Train the model on the subset of the training data
            
            # Stop timer and record time
            elapsed_time = time.time() - start_time  # Calculate how long the training took
            times.append(elapsed_time)  # Store the training time
            
            # Predict on test set and calculate accuracy
            y_pred = model.predict(X_test)  # Use the trained model to predict labels on the test set
            accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy of predictions
            accuracies.append(accuracy)  # Store the accuracy

            # Print results
            print(f"Dataset Size: {size}")
            print(f"Training Time: {elapsed_time:.4f} seconds")
            print(f"Accuracy: {accuracy:.4f}")
            print("----------------------------------------")
            
        except Exception as e:
            # In case the model completely fails, record NaN for time and accuracy
            times.append(float('nan'))  # Append NaN for time if the model fails
            accuracies.append(float('nan'))  # Append NaN for accuracy if the model fails
            print(f"Failed to converge for size {size}: {e}")  # Print error message

    # Print warnings, if any
    for warning in w:
        print(f"Warning: {warning.message}")  # Print any warnings captured during model training

# Step 3: Visualize the training times for different dataset sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, times, marker='o', color='b')  # Plot training time for each dataset size
plt.title('Logistic Regression Training Time vs Dataset Size (Struggling Simulation)')  # Title of the plot
plt.xlabel('Dataset Size')  # Label for x-axis (dataset size)
plt.ylabel('Training Time (seconds)')  # Label for y-axis (training time)
plt.grid(True)  # Add a grid to the plot
plt.show()  # Display the training time plot

# Step 4: Visualize the accuracies for different dataset sizes
plt.figure(figsize=(10, 6))
plt.plot(data_sizes, accuracies, marker='o', color='r')  # Plot accuracy for each dataset size
plt.title('Logistic Regression Accuracy vs Dataset Size (Struggling Simulation)')  # Title of the plot
plt.xlabel('Dataset Size')  # Label for x-axis (dataset size)
plt.ylabel('Accuracy')  # Label for y-axis (accuracy score)
plt.grid(True)  # Add a grid to the plot
plt.show()  # Display the accuracy plot
