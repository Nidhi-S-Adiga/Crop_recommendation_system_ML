import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Crop_recommendation.csv')

# Check dataset info
print(df.info())

# Check unique labels
print(df['label'].unique())
# Check for missing values
print(df.isnull().sum())
# Summary statistics
print(df.describe())
# Encode categorical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Save the label encoder
with open('LabelEncoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label to numeric value mapping:")
for numeric_value, original_label in label_mapping.items():
    print(f"{original_label}: {numeric_value}")


# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['label']))

# Save the scaler
with open('Scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# PCA Transformation
pca = PCA(n_components=0.95)  # Preserve 95% of the variance
X_pca = pca.fit_transform(scaled_features)
# Save the PCA model
with open('PCA.pkl', 'wb') as file:
    pickle.dump(pca, file)

scaled_pca_df = pd.DataFrame(X_pca)
scaled_pca_df['label'] = df['label']


# Split the Data
X = scaled_pca_df.drop(columns=['label'])
y = scaled_pca_df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display shapes of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

acc = []
model = []
from sklearn.tree import plot_tree
# Decision Tree
DecisionTree = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=10)
DecisionTree.fit(X_train, y_train)
predicted_values = DecisionTree.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')
print("Decision Tree's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))

#plt.figure(figsize=(15, 10))  # Set the figure size
#plot_tree(DecisionTree)
#plt.title("Decision Tree (Entropy Criterion)")
#plt.show()

plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), DecisionTree.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
plt.show()
# Decision Tree with Gini impurity
DecisionTree_gini = DecisionTreeClassifier(criterion="gini", random_state=2, max_depth=10)
DecisionTree_gini.fit(X_train, y_train)
predicted_values_gini = DecisionTree_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, predicted_values_gini)
acc.append(accuracy_gini)
model.append('Decision Tree (Gini)')
print("Decision Tree (Gini) Accuracy is: ", accuracy_gini * 100)
print(classification_report(y_test, predicted_values_gini))

plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), DecisionTree_gini.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
plt.show()

# Naive Bayes
NaiveBayes = GaussianNB()
NaiveBayes.fit(X_train, y_train)
predicted_values = NaiveBayes.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Naive Bayes')
print("Naive Bayes's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# SVM
SVM = SVC(gamma='auto')
SVM.fit(X_train, y_train)
predicted_values = SVM.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('SVM')
print("SVM's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# Logistic Regression
LogReg = LogisticRegression(random_state=2)
LogReg.fit(X_train, y_train)
predicted_values = LogReg.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Logistic Regression')
print("Logistic Regression's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# Random Forest
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train, y_train)
predicted_values = RF.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Random Forest')
print("Random Forest's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# XGBoost
XB = xgb.XGBClassifier()
XB.fit(X_train, y_train)
predicted_values = XB.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('XGBoost')
print("XGBoost's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# Bagging
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
predicted_values = bagging.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Bagging')
print("Bagging's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# Boosting
boosting = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
boosting.fit(X_train, y_train)
predicted_values = boosting.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Boosting')
print("Boosting's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
# Voting
voting = VotingClassifier(estimators=[
    ('dt', DecisionTreeClassifier()),
    ('nb', GaussianNB()),
    ('svm', SVC(gamma='auto', probability=True)),
    ('logreg', LogisticRegression(random_state=2)),
    ('rf', RandomForestClassifier(n_estimators=20, random_state=0)),
    ('xgb', xgb.XGBClassifier())
], voting='soft')
voting.fit(X_train, y_train)
predicted_values = voting.predict(X_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Voting')
print("Voting's Accuracy is: ", x*100)
print(classification_report(y_test, predicted_values))
plt.figure(figsize=[10,5], dpi=100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=model, palette='dark')

accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print(f"{k} --> {v}")

models = {
    'Decision Tree': DecisionTree,
    'Naive Bayes': NaiveBayes,
    'SVM': SVM,
    'Logistic Regression': LogReg,
    'Random Forest': RF,
    'XGBoost': XB,
    'Bagging': bagging,
    'Boosting': boosting,
    'Voting': voting
}

# Save models using pickle
for model_name, model_instance in models.items():
    filename = f"{model_name.replace(' ', '_')}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model_instance, file)


def predict_new_data(data, scaler, pca, model, label_encoder):
    # Scale the data
    scaled_data = scaler.transform(data)

    # Apply PCA
    pca_data = pca.transform(scaled_data)

    # Predict the encoded label
    encoded_prediction = model.predict(pca_data)

    # Convert the encoded label back to the original categorical label
    categorical_prediction = label_encoder.inverse_transform(encoded_prediction)

    return categorical_prediction[0]

# Example usage
new_data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
new_data_df = pd.DataFrame(new_data, columns=feature_names)

# Load models and encoders
with open('Scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
with open('PCA.pkl', 'rb') as file:
    loaded_pca = pickle.load(file)
with open('LabelEncoder.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)
with open('Voting.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

predicted_label = predict_new_data(new_data_df, loaded_scaler, loaded_pca, loaded_model, loaded_label_encoder)
print(f'Predicted Categorical Label: {predicted_label}')

