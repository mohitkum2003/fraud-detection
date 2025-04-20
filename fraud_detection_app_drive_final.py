
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Fraud Detection", layout="wide")

st.title("💳 Online Fraud Detection Dashboard")

# Load CSV from Google Drive
file_id = "1HVB5v-IyppmTznakjE7v3kfsbXG84-ug"  # Replace with actual Google Drive file ID
dwn_url = f"https://drive.google.com/uc?id={file_id}"

@st.cache_data
def load_data():
    df = pd.read_csv(dwn_url)
    return df

dataframe = load_data()

st.success("✅ Dataset loaded successfully from Google Drive!")

st.subheader("🔍 Data Preview")
st.write(dataframe.head())

st.subheader("📊 Class Distribution")
if 'isFraud' in dataframe.columns:
    class_counts = dataframe['isFraud'].value_counts()
    st.bar_chart(class_counts)

st.subheader("📈 Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(dataframe.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("📌 Dataset Info")
st.write(dataframe.describe())
st.write("Missing Values:", dataframe.isnull().sum())

# Original notebook code (unchanged)
st.markdown("## 🔧 Additional Notebook Code")
with st.expander("Show Raw Code"):
    st.code("""import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataframe=pd.read_csv("PS_20174392719_1491204439457_log.csv")

dataframe.head()

# Checking for null value
dataframe.isnull()

 # Methods to get columns in dataset

dataframe.columns

dataframe.columns.values

list(dataframe)

# Getting to know about data
dataframe.info

dataframe.describe()

col = list(dataframe)

col

dataframe.isnull().sum()

# Checking for duplicate values
dataframe.duplicated()

dataframe.duplicated().sum()

# Univariate Analysis

# Distribution of Transaction money
plt.hist(dataframe['amount'],bins=20)
plt.xlabel('amount')
plt.ylabel('count')
plt.title("distribution of transaction amounts")

dataframe['isFraud'].value_counts()
#this will tell that in a column , what is the frequency of occurance of 1 and 0
#0 = not fraud; 1 = isfraud

sns.countplot(x='isFraud',data=dataframe)
#this will show how much isFraud col is divided into its categories
#count shows count of occurances
plt.show()

#now break down transaction's that has been done
#as done above for isFraud col, do same for type of transactions col, and find occurances of each category

dataframe["type"].values #values of type col

dataframe["type"].value_counts()

sns.countplot(x="type",data=dataframe)

dataframe['type'].value_counts().index #list of unique values in decresing order

sns.countplot(x='type',data=dataframe,order=dataframe['type'].value_counts().index)

#bi & multi variate

#heat map is used to get correlation b/w each numeric variable to each other
#we use this for quickly identify strong and weak correlations,dealing with large datasets,detect multi collinearity
correlation=dataframe.select_dtypes('number').corr()
sns.heatmap(correlation,cmap='coolwarm',annot=True)
plt.title('heatmap of correlation matrix')
plt.show()

list(dataframe)

#boxplot used to distribute conti. variable acc. to different categories, isfraud is category and transaction ammount is conti. variable
sns.boxplot(x='isFraud',y='amount',data=dataframe,hue='type')
plt.xlabel('isFraud')
plt.ylabel('transaction amount')
plt.title('amount for fraud transaction')
plt.show()
#this graph will show that cash_out and transfer are the commonly used payment types
#used by scammers

#one hot encoding
encoded_data=pd.get_dummies(dataframe,columns=['type'],prefix='type')
encoded_data.head()

list(encoded_data)

target_var=dataframe['isFraud'] #select target variable
#target varible is what we aim to predict

target_var

#select feature columns
feature_col=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
feature_var=encoded_data[feature_col]
#feature variable are attributes of data that we use to make predictions

feature_var.head()

dataframe['isFraud'].value_counts()

# Store Feature Matrix In XX And Response (Target) In Vector YY
XX = dataframe.drop('isFraud',axis=1)
YY = dataframe['isFraud']
normal = dataframe[dataframe['isFraud']==0]
fraud = dataframe[dataframe['isFraud']==1]

normal.shape

fraud.shape

normal_sample=normal.sample(n=8213)

normal_sample.shape

new_data = pd.concat([normal_sample,fraud],ignore_index=True)

new_data['isFraud'].value_counts()

new_data.head()

#one hot encoding
encoded_data=pd.get_dummies(new_data,columns=['type'],prefix='type')
encoded_data.head()

# XX = new_data.drop('isFraud',axis=1)
feature_col=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_IN','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']
feature_var=encoded_data[feature_col]
target_var = new_data['isFraud']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(feature_var,target_var,test_size=0.20,random_state=42)

print('X_train',X_train.shape)
print('X_test',X_test.shape)
print('Y_train',Y_train.shape)
print('Y_test',Y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score,f1_score

random_forest_model=RandomForestClassifier()

random_forest_model.fit(X_train,Y_train)

random_forest_pred=random_forest_model.predict(X_test)

accuracy_score(Y_test,random_forest_pred)

precision_score(Y_test,random_forest_pred)

from sklearn.metrics import precision_score,recall_score,f1_score

recall_score(Y_test,random_forest_pred)

f1_score(Y_test,random_forest_pred)

#getting accuracy, precision f1 score for random forest
random_f_accuracy=accuracy_score(Y_test,random_forest_pred)*100
random_f_precision=precision_score(Y_test,random_forest_pred)*100
random_f_f1=f1_score(Y_test,random_forest_pred)*100
random_f_recall=recall_score(Y_test,random_forest_pred)*100
print("Random Forest")
print("Accuracy",random_f_accuracy)
print("Precision",random_f_precision)
print("F1-score",random_f_f1)
print("Recall",random_f_recall)
print()

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)

dt_pred=dt.predict(X_test)

#getting accuracy, precision f1 score for random forest
tree_f_accuracy=accuracy_score(Y_test,dt_pred)*100
tree_f_precision=precision_score(Y_test,dt_pred)*100
tree_f_f1=f1_score(Y_test,dt_pred)*100
tree_f_recall=recall_score(Y_test,dt_pred)*100
print("Decision Tree")
print("Accuracy",tree_f_accuracy)
print("Precision",tree_f_precision)
print("F1-score",tree_f_f1)
print("Recall",tree_f_recall)
print()

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)  # Increase max_iter
logistic_model.fit(X_train, Y_train)

log_pred=logistic_model.predict(X_test)

#getting accuracy, precision f1 score for logistic regression
logistic_accuracy=accuracy_score(Y_test,log_pred)*100
logistic_precision=precision_score(Y_test,log_pred)*100
logistic_f1=f1_score(Y_test,log_pred)*100 # Now, f1_score refers to the function
logistic_recall=recall_score(Y_test,log_pred)*100
print("logistic regression")
print("Accuracy",logistic_accuracy)
print("Precision",logistic_precision)
print("F1-score",logistic_f1)
print("Recall",logistic_recall)
print()

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
              "ACC":[logistic_accuracy,
                     tree_f_accuracy,
                     random_f_accuracy],
              "PRE":[logistic_precision,
                     tree_f_precision,
                     random_f_precision],
              "F1":[logistic_f1,
                    tree_f_f1,
                     random_f_f1],
              "Recall":[logistic_recall,tree_f_recall,random_f_recall]})

final_data

# sns.barplot(x='Models', y=['ACC','PRE','F1','Recall'], data=final_data)
# Reshape the 'final_data' DataFrame to a long format
final_data_melted = pd.melt(final_data, id_vars=['Models'], value_vars=['ACC', 'PRE', 'F1', 'Recall'],
                           var_name='Metric', value_name='Score')

# Now you can create the barplot using the melted DataFrame
sns.barplot(x='Models', y='Score', hue='Metric', data=final_data_melted)
plt.show()

#Voting (hard Voting)
from sklearn.ensemble import VotingClassifier
# Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(estimators=[('LR', logistic_model), ('DT', dt), ('RF', random_forest_model)], voting='hard')

# Train the ensemble model
voting_clf.fit(X_train, Y_train)

# Predictions
vote_pred = voting_clf.predict(X_test)

# Evaluate Performance
vote_accuracy = accuracy_score(Y_test, vote_pred)
print("Voting Model Accuracy:",vote_accuracy * 100)

#getting accuracy, precision f1 score for voting_clf
vote_accuracy=accuracy_score(Y_test,vote_pred)*100
vote_precision=precision_score(Y_test,vote_pred)*100
vote_f1=f1_score(Y_test,vote_pred)*100
vote_recall=recall_score(Y_test,vote_pred)*100
print("Voting Classifier")
print("Accuracy",vote_accuracy)
print("Precision",vote_precision)
print("F1-score",vote_f1)
print("Recall",vote_recall)
print()


#Stacking
from sklearn.ensemble import StackingClassifier

# Define Base Models
base_models = [
    ('RF', RandomForestClassifier()),
    ('DT', DecisionTreeClassifier()),
    ('LR', LogisticRegression(max_iter=1000))
]

# Meta-Classifer (Final Decision Maker)
meta_model = RandomForestClassifier()

# Stacking Classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Train Model
stacking_clf.fit(X_train, Y_train)

# Predictions
stack_pred = stacking_clf.predict(X_test)

#getting accuracy, precision f1 score for voting_clf
stack_accuracy=accuracy_score(Y_test,stack_pred)*100
stack_precision=precision_score(Y_test,stack_pred)*100
stack_f1=f1_score(Y_test,stack_pred)*100
stack_recall=recall_score(Y_test,stack_pred)*100
print("Voting Classifier")
print("Accuracy",stack_accuracy)
print("Precision",stack_precision)
print("F1-score",stack_f1)
print("Recall",stack_recall)
print()


# Predict probabilities
rf_probs = random_forest_model.predict_proba(X_test)
dt_probs = dt.predict_proba(X_test)
lr_probs = logistic_model.predict_proba(X_test)

# Simple averaging of probabilities
avg_probs = (rf_probs + dt_probs + lr_probs) / 3

# Convert averaged probabilities to class predictions
avg_pred = np.argmax(avg_probs, axis=1)

# Evaluate
# Getting accuracy, precision, F1 score for avg_clf
avg_accuracy = accuracy_score(Y_test, avg_pred) * 100
avg_precision = precision_score(Y_test, avg_pred) * 100
avg_f1 = f1_score(Y_test, avg_pred) * 100
avg_recall = recall_score(Y_test, avg_pred) * 100

print("Voting Classifier")
print("Accuracy:", avg_accuracy)
print("Precision:", avg_precision)
print("F1-score:", avg_f1)
print("Recall:", avg_recall)
print()



import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

# Predict probabilities
rf_probs = random_forest_model.predict_proba(X_test)
dt_probs = dt.predict_proba(X_test)
lr_probs = logistic_model.predict_proba(X_test)

# Assign weights (Random Forest > Decision Tree > Logistic Regression)
weights = np.array([0.5, 0.3, 0.2])  # Weights for [rf, dt, lr]

# Weighted averaging of probabilities
weighted_probs = (weights[0] * rf_probs) + (weights[1] * dt_probs) + (weights[2] * lr_probs)


# Convert probabilities to final predictions
final_predictions_weighted = np.argmax(weighted_probs, axis=1)

# Evaluate
weighted_accuracy = accuracy_score(Y_test, final_predictions_weighted) * 100
weighted_precision = precision_score(Y_test, final_predictions_weighted) * 100
weighted_f1 = f1_score(Y_test, final_predictions_weighted) * 100
weighted_recall = recall_score(Y_test, final_predictions_weighted) * 100

print("Weighted Averaging Classifier")
print("Accuracy:", weighted_accuracy)
print("Precision:", weighted_precision)
print("F1-score:", weighted_f1)
print("Recall:", weighted_recall)


models_tech_data = pd.DataFrame({'Techniques':['Voting','Stacking','Averaging', 'Weighted avg'],
              "ACC":[vote_accuracy,
                     stack_accuracy,
                     avg_accuracy,weighted_accuracy],
              "PRE":[vote_precision,
                     stack_precision,
                     avg_precision,weighted_precision],
              "F1":[vote_f1,
                    stack_f1,
                     avg_f1,weighted_f1],
              "Recall":[vote_recall,stack_recall,avg_recall,weighted_recall]})

models_tech_data

# sns.barplot(x='Techniques', y=['ACC','PRE','F1','Recall'], data=model_tech__data)
# Reshape the 'final_data' DataFrame to a long format
technique_final_data_melted = pd.melt(models_tech_data, id_vars=['Techniques'], value_vars=['ACC', 'PRE', 'F1', 'Recall'],
                           var_name='Metric', value_name='Score')

# Now you can create the barplot using the melted DataFrame
sns.barplot(x='Techniques', y='Score', hue='Metric', data=technique_final_data_melted) # Use the melted DataFrame 'technique_final_data_melted'
plt.show()

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights):
        """
        Initialize the weighted ensemble classifier.

        Parameters:
        models -- List of trained classifier models
        weights -- Array of weights for each model (should sum to 1)
        """
        self.models = models
        self.weights = weights

    def predict_proba(self, X):
        """
        Predict class probabilities for X using weighted average of probabilities.
        """
        probas = [model.predict_proba(X) for model in self.models]
        weighted_probas = np.zeros_like(probas[0])

        for i, proba in enumerate(probas):
            weighted_probas += self.weights[i] * proba

        return weighted_probas

    def predict(self, X):
        """
        Predict class labels for X.
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Create the weighted ensemble classifier
models = [random_forest_model, dt, logistic_model]
weights = np.array([0.5, 0.3, 0.2])  # Same weights you used earlier

ensemble_model = WeightedEnsembleClassifier(models=models, weights=weights)

# Save the ensemble model
joblib.dump(ensemble_model, 'fraud_detection_ensemble.pkl')

# If you want to also save individual models for future use:
joblib.dump(random_forest_model, 'random_forest_model.pkl')
joblib.dump(dt, 'decision_tree_model.pkl')
joblib.dump(logistic_model, 'logistic_model.pkl')

print("Model saved successfully!")

def predict_fraud(transaction_data, model_path='fraud_detection_ensemble.pkl'):
    """
    Predict whether a transaction is fraudulent.

    Parameters:
    transaction_data -- DataFrame containing transaction features
    model_path -- Path to the saved model

    Returns:
    prediction -- 0 for legitimate transaction, 1 for fraudulent
    probability -- Probability of fraud
    """
    # Load the saved model
    model = joblib.load(model_path)

    # Make prediction
    prediction = model.predict(transaction_data)[0]

    # Get probability of fraud (class 1)
    probability = model.predict_proba(transaction_data)[0][1]

    return prediction, probability

def preprocess_transaction(transaction, scaler=None):
    """
    Preprocess a single transaction for prediction.

    Parameters:
    transaction -- Dictionary containing transaction features
    scaler -- Fitted scaler object (if you used scaling in training)

    Returns:
    processed_data -- DataFrame ready for prediction
    """
    import pandas as pd

    # Expected feature columns
    expected_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
                        'newbalanceDest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                        'type_PAYMENT', 'type_TRANSFER']

    # Create a DataFrame with a single row
    transaction_df = pd.DataFrame([transaction])

    # Check if all expected columns are present
    for col in expected_columns:
        if col not in transaction_df.columns:
            transaction_df[col] = 0  # Default to 0 if missing

    # Ensure proper column order
    transaction_df = transaction_df[expected_columns]

    # Apply scaling if provided and used during training
    if scaler is not None:
        numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        transaction_df[numeric_cols] = scaler.transform(transaction_df[numeric_cols])

    return transaction_df

def fraud_detection_pipeline(transaction, model_path='fraud_detection_ensemble.pkl', scaler_path=None):
    """
    Complete pipeline for fraud detection.

    Parameters:
    transaction -- Dictionary with transaction features
    model_path -- Path to the saved model
    scaler_path -- Path to the saved scaler (if used)

    Returns:
    result -- Dictionary with prediction result and probability
    """
    # Load scaler if path provided
    scaler = None
    if scaler_path:
        scaler = joblib.load(scaler_path)

    # Preprocess transaction
    processed_data = preprocess_transaction(transaction, scaler)

    # Make prediction
    prediction, probability = predict_fraud(processed_data, model_path)

    # Create result dictionary
    result = {
        'is_fraud': bool(prediction),
        'fraud_probability': float(probability),
        'fraud_confidence': 'High' if probability > 0.75 else 'Medium' if probability > 0.5 else 'Low'
    }

    return result

# Example transaction
new_transaction = {
        'amount': 2000.0,
        'oldbalanceOrg': 10000.0,
        'newbalanceOrig': 8000.0,  # Gradual decrease, not suspicious
        'oldbalanceDest': 7000.0,
        'newbalanceDest': 9000.0,  # Normal deposit
        'type_CASH_IN': 0,
        'type_CASH_OUT': 0,
        'type_DEBIT': 0,
        'type_PAYMENT': 0,
        'type_TRANSFER': 1  # Small transfer, common transaction
    },

# If you used a scaler during training
# scaler_path = 'your_scaler.pkl'  # Path to your saved scaler
# result = fraud_detection_pipeline(new_transaction, scaler_path=scaler_path)

# If no scaler was used
result = fraud_detection_pipeline(new_transaction)

print("Transaction analysis:")
print(f"Is fraud: {result['is_fraud']}")
print(f"Fraud probability: {result['fraud_probability']:.2%}")
print(f"Confidence: {result['fraud_confidence']}")""", language='python')
