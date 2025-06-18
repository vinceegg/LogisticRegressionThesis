import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('datasets/email.csv')  # Reading the CSV file containing email data

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Category'], test_size=0.2, random_state=42) 
# Splitting the dataset into training (80%) and testing (20%) sets

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  # Creating a vectorizer to convert text to numeric counts, removing English stop words
X_train_counts = vectorizer.fit_transform(X_train)  # Fit the vectorizer on the training data and transform the text into count vectors
X_test_counts = vectorizer.transform(X_test)  # Transform the test data using the same vectorizer

# Train a logistic regression model with more bias towards 'ham'
logreg = LogisticRegression(max_iter=1000, class_weight={'ham': 10, 'spam': 1})  
# Logistic regression model that increases the importance of 'ham' class by assigning higher weight (10) compared to 'spam' (1)
logreg.fit(X_train_counts, y_train)  # Training the model on the vectorized training data

# Predictions on the test set
y_pred = logreg.predict(X_test_counts)  # Predicting labels (ham/spam) for the test set
y_pred_prob = logreg.predict_proba(X_test_counts)[:, 1]  # Predicting the probabilities of the positive class (spam)

# Show the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))  # Printing precision, recall, f1-score, and support for each class

# Show the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))  # Printing the confusion matrix to see true positives, false positives, etc.

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test.map({'ham': 0, 'spam': 1}), y_pred_prob)  
# Calculate False Positive Rate (FPR) and True Positive Rate (TPR) for different thresholds
roc_auc = auc(fpr, tpr)  # Compute the Area Under the ROC Curve (AUC)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')  
# Plotting the ROC curve with the calculated AUC value
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Adding a diagonal line for random guessing
plt.xlim([0.0, 1.0])  # Setting the x-axis limits
plt.ylim([0.0, 1.05])  # Setting the y-axis limits
plt.xlabel('False Positive Rate')  # Label for the x-axis
plt.ylabel('True Positive Rate')  # Label for the y-axis
plt.title('Receiver Operating Characteristic (ROC) Curve')  # Title of the plot
plt.legend(loc="lower right")  # Placing the legend in the lower-right corner
plt.show()  # Display the ROC curve plot

# Histogram of predicted probabilities for spam (red) and ham (blue)
plt.figure(figsize=(10, 6))

# Plot ham probabilities (0) in blue
sns.histplot(y_pred_prob[y_test == 'ham'], bins=30, kde=False, color='blue', label='Ham')  
# Plot a histogram of predicted probabilities for ham emails (ground truth = 'ham')

# Plot spam probabilities (1) in red
sns.histplot(y_pred_prob[y_test == 'spam'], bins=30, kde=False, color='red', label='Spam')  
# Plot a histogram of predicted probabilities for spam emails (ground truth = 'spam')

plt.title('Histogram of Predicted Probabilities')  # Title of the histogram plot
plt.xlabel('Predicted Probability')  # Label for the x-axis (predicted probability)
plt.ylabel('Frequency')  # Label for the y-axis (frequency of each probability bin)
plt.legend()  # Display the legend
plt.show()  # Show the histogram plot
