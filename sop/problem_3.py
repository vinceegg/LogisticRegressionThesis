import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('datasets/email_datasets.csv')  # Load the email dataset

# Encode 'Category' column to binary (spam=1, ham=0)
label_encoder = LabelEncoder()  # Create a label encoder to convert categorical labels into numeric form
data['Category_encoded'] = label_encoder.fit_transform(data['Category'])  # Encode 'Category' into binary values

# Convert text data into Bag of Words (without removing stopwords or limiting features)
count_vectorizer = CountVectorizer(stop_words=None, max_features=None)  # Create a CountVectorizer without restrictions
X = count_vectorizer.fit_transform(data['Message'])  # Transform the 'Message' text into numerical feature vectors
y = data['Category_encoded']  # Target variable (encoded categories)

# Reduce the training set size (use only 10% of the data for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)  
# Split the data, using only 10% for training and 90% for testing

# Train a logistic regression model without regularization (C=1e10 to remove regularization)
log_reg = LogisticRegression(C=1e10, max_iter=1000)  # Set C to a high value to remove regularization
log_reg.fit(X_train, y_train)  # Train the logistic regression model on the training set

# Predict on training data and testing data
y_train_pred = log_reg.predict(X_train)  # Predict labels on the training set
y_test_pred = log_reg.predict(X_test)  # Predict labels on the test set

# Calculate accuracy for both train and test data
train_accuracy = accuracy_score(y_train, y_train_pred)  # Calculate accuracy for the training set
test_accuracy = accuracy_score(y_test, y_test_pred)  # Calculate accuracy for the test set

# Visualize the results in a bar graph
accuracy_values = [train_accuracy, test_accuracy]  # Store accuracy values for both training and testing
labels = ['Train Accuracy', 'Test Accuracy']  # Labels for the bar graph

plt.bar(labels, accuracy_values, color=['blue', 'green'])  # Plot a bar graph of accuracies
plt.ylim([0, 1])  # Set the y-axis range from 0 to 1
plt.title('Logistic Regression: Train vs Test Accuracy')  # Title of the bar graph
plt.ylabel('Accuracy')  # Label for the y-axis (accuracy)
plt.show()  # Display the plot

# Print the accuracy for both train and test sets
print(f'Train Accuracy: {train_accuracy}')  # Print training accuracy
print(f'Test Accuracy: {test_accuracy}')  # Print test accuracy
