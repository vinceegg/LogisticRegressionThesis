# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('datasets/email_datasets.csv')  # Load the email dataset

# Encode 'Category' column to binary (spam=1, ham=0)
label_encoder = LabelEncoder()
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])  # Encode the labels (spam as 1, ham as 0)

# Convert text data into TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)  # Limit the number of features to 3000
X = tfidf_vectorizer.fit_transform(data['Message']).toarray()  # Convert text to numerical features using TF-IDF
y = data['Category_encoded']  # Target labels

# Apply PCA (Principal Component Analysis) to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% of the variance with the reduced components
X_pca = pca.fit_transform(X)  # Transform the TF-IDF features into reduced dimensions using PCA

# Split the data into training and testing sets (use only 10% of the data for training)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.9, random_state=42)  # 90% for testing, 10% for training

# Train a logistic regression model (with regularization)
log_reg = LogisticRegression(C=1.0, max_iter=1000)  # Use regularization (C=1.0) to prevent overfitting
log_reg.fit(X_train, y_train)  # Fit the model to the training data

# Predict on training data and testing data
y_train_pred = log_reg.predict(X_train)  # Predict on the training set
y_test_pred = log_reg.predict(X_test)  # Predict on the testing set

# Calculate accuracy for both train and test data
train_accuracy = accuracy_score(y_train, y_train_pred)  # Accuracy on training data
test_accuracy = accuracy_score(y_test, y_test_pred)  # Accuracy on testing data

# Visualize the results in a bar graph
accuracy_values = [train_accuracy, test_accuracy]  # List of accuracy values for training and testing
labels = ['Train Accuracy', 'Test Accuracy']  # Labels for the bar chart

plt.bar(labels, accuracy_values, color=['blue', 'green'])  # Create a bar plot for train and test accuracy
plt.ylim([0, 1])  # Set the y-axis limits from 0 to 1 (percentage)
plt.title('Logistic Regression with TF-IDF & PCA: Train vs Test Accuracy')  # Set plot title
plt.ylabel('Accuracy')  # Label for the y-axis
plt.show()  # Display the plot

# Print the train and test accuracies
print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
