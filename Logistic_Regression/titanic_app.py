import streamlit as st
import pandas as pd
import joblib

# Load trained model and scaler
model = joblib.load('titanic_logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title(" Titanic Survival Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a Titanic CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    st.success(" File successfully uploaded!")

    # Read CSV
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Raw Uploaded Data")
    st.write(test_df.head())

    # Drop unnecessary columns
    test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

    # Fill missing values
    test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
    test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
    test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

    # Encoding categorical variables using get_dummies like in training
    test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

    # Define expected columns in correct order as per model training
    expected_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                        'Sex_male', 'Embarked_Q', 'Embarked_S']

    # Add any missing columns with value 0 (if, say, no 'Embarked_Q' in this batch)
    for col in expected_columns:
        if col not in test_df.columns:
            test_df[col] = 0

    # Reorder columns to match the training set
    X_test_final = test_df[expected_columns]

    X_test_scaled = X_test_final.copy()
    
    num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    ## Scale
    X_test_scaled[num_cols] = scaler.fit_transform(X_test_final[num_cols])
    

    # Predict
    predictions = model.predict(X_test_scaled)
    prediction_probs = model.predict_proba(X_test_scaled)

    # Attach predictions to DataFrame
    test_df['Predicted_Survival'] = predictions
    test_df['Survival_Probability'] = prediction_probs[:, 1]

    # Show predictions
    st.subheader("Predictions")
    st.write(test_df[['Pclass', 'Age', 'Fare', 'Predicted_Survival', 'Survival_Probability']])

    # Option to download result
    csv = test_df.to_csv(index=False)
    st.download_button(" Download Predictions as CSV", data=csv,
                       file_name="titanic_predictions.csv", mime='text/csv')

else:
    st.info(" Upload a CSV file above to get predictions.")
