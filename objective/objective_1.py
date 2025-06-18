import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('datasets/email.csv')  # Load the email dataset

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42)  
# Split the data into training and testing sets (80% train, 20% test)

# Vectorize the text using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # Create a TF-IDF vectorizer to convert text into numerical format
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform the training data into TF-IDF features
X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Transform the test data using the same TF-IDF vectorizer

# Train a logistic regression model with balanced class weights
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')  # Use class_weight='balanced' to handle class imbalance
logreg.fit(X_train_tfidf, y_train)  # Train the logistic regression model on the TF-IDF-transformed training data

# Predictions on the test set
y_pred = logreg.predict(X_test_tfidf)  # Predict labels on the test set
y_pred_prob = logreg.predict_proba(X_test_tfidf)[:, 1]  # Get predicted probabilities for the positive class (spam)

# Show the classification report
print("Classification Report:")  # Print the classification report (precision, recall, f1-score)
print(classification_report(y_test, y_pred))

# Show the confusion matrix
print("Confusion Matrix:")  # Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test.map({'ham': 0, 'spam': 1}), y_pred_prob)  
# Compute false positive rate (FPR) and true positive rate (TPR) for the ROC curve
roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve (AUC)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')  # Plot the ROC curve
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Plot the diagonal (random classifier)
plt.xlim([0.0, 1.0])  # Set x-axis limits
plt.ylim([0.0, 1.05])  # Set y-axis limits
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Receiver Operating Characteristic (ROC) Curve')  # Title of the plot
plt.legend(loc="lower right")  # Add legend to the plot
plt.show()  # Display the ROC curve plot

# Histogram of predicted probabilities for spam (red) and ham (blue)
plt.figure(figsize=(10, 6))

# Plot ham probabilities (0) in blue
sns.histplot(y_pred_prob[y_test == 'ham'], bins=30, kde=False, color='blue', label='Ham')  
# Plot histogram of predicted probabilities for ham emails

# Plot spam probabilities (1) in red
sns.histplot(y_pred_prob[y_test == 'spam'], bins=30, kde=False, color='red', label='Spam')  
# Plot histogram of predicted probabilities for spam emails

plt.title('Histogram of Predicted Probabilities')  # Title of the histogram
plt.xlabel('Predicted Probability')  # Label for the x-axis (predicted probability)
plt.ylabel('Frequency')  # Label for the y-axis (frequency)
plt.legend()  # Add a legend to distinguish ham and spam
plt.show()  # Display the histogram plot
