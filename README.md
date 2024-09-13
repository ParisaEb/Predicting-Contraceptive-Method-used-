# Contraceptive Method Prediction and Clustering Analysis
This project explores predictive modeling and clustering analysis using a dataset related to contraceptive method usage. It involves classification, model evaluation, and clustering techniques to extract insights from the data.

# Project Overview
The project is divided into several key components:

Predictive Modeling: Using machine learning algorithms like Random Forest and Naive Bayes to predict the contraceptive method used based on various features.
Model Evaluation: Evaluating model performance using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.
Clustering Analysis: Applying clustering techniques like K-Means and Agglomerative Clustering to group similar instances in the dataset.
Dimensionality Reduction: Utilizing PCA to reduce the dimensionality of the dataset for visualization and analysis.
Prerequisites
Ensure you have the following Python packages installed to run the project:

Python 3.6 or higher
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
You can install the required packages using the following command:

bash
Copy code
pip install pandas numpy scikit-learn matplotlib seaborn
Dataset
The dataset used in this project is a contraceptive method choice dataset, which includes various features such as:

Wife's age
Education level (Wife's and Husband's)
Number of children ever born
Standard of living index
Media exposure
Contraceptive method used
The dataset is loaded from a CSV file and is preprocessed for further analysis.

Predictive Modeling
Random Forest Classifier
A Random Forest classifier is trained to predict the contraceptive method used based on the features. Hyperparameter tuning is performed using GridSearchCV to find the best model parameters.

Example:

python
Copy code
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest and GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)
Naive Bayes Classifier
A Gaussian Naive Bayes classifier is also implemented to provide a baseline for comparison. The model is evaluated using accuracy, precision, recall, and F1-score.

Example:

python
Copy code
from sklearn.naive_bayes import GaussianNB

# Initialize and train the model
model = GaussianNB()
model.fit(X_train_scaled, y_train)
Model Evaluation
The performance of the models is evaluated using:

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Example:

python
Copy code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"Confusion matrix:\n{conf_matrix}")
Clustering Analysis
K-Means Clustering
K-Means clustering is applied to group instances in the dataset into clusters. The elbow method is used to determine the optimal number of clusters, and the silhouette score is calculated to evaluate the clustering quality.

Example:

python
Copy code
from sklearn.cluster import KMeans

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Calculate silhouette score
silhouette = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette}")
Agglomerative Clustering
Agglomerative Clustering is also applied, and the results are visualized using a dendrogram.

Example:

python
Copy code
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Perform Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='complete')
agg.fit(X_scaled)

# Plot dendrogram
Z = linkage(X_scaled, method='complete')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.show()
Visualization
The results of both the classification and clustering models are visualized using various plots, including:

Confusion Matrix
Cluster Scatter Plots
Dendrogram
Elbow Method Plot
License
This project is licensed under the MIT License. See the LICENSE file for details.

