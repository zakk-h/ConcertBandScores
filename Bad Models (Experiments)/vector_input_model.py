import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def standardize_school_name(school_name):
    if pd.isna(school_name):
        return ''
    
    priority_words = ['symphonic', 'wind', 'concert', '7th', '8th', 'middle', 'high']
    parts = school_name.lower().split()
    
    # Extract the first word
    first_word = parts[0]
    
    # Extract the second word if it's not a priority word
    second_word = parts[1] if len(parts) > 1 and parts[1] not in priority_words else ''
    
    # Find the highest priority word in the name
    highest_priority_word = ''
    for word in priority_words:
        if word in parts:
            highest_priority_word = word
            break
    
    # Create the standardized name
    standardized_name = first_word
    if second_word:
        standardized_name += f" {second_word}"
    if highest_priority_word:
        standardized_name += f" {highest_priority_word.capitalize()}"
    
    return standardized_name

def preprocess_data(df, label_encoders=None, fit_encoders=False):
    # Define a one-to-one mapping from roman numerals to integers
    roman_to_int = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'I/II': 1.5,  # In-between scores
        'II/III': 2.5,
        'III/IV': 3.5,
        'IV/V': 4.5,
        'V/VI': 5.5,
        'VI/MW': 6.5,  # VI/MW treated as 6.5
        'MW': 7,  # MW treated as 7
        'co': None,  # Comments only
        'C/O': None  # Comments only
    }
    
    # Standardize school names
    df['School'] = df['School'].apply(standardize_school_name)
    
    # Replace Roman numerals with integers
    df['Judge 1'] = df['Judge 1'].map(roman_to_int)
    df['Judge 2'] = df['Judge 2'].map(roman_to_int)
    df['Judge 3'] = df['Judge 3'].map(roman_to_int)
    
    # Calculate the total score across all judges, lower is better
    df['Total Score'] = df[['Judge 1', 'Judge 2', 'Judge 3']].sum(axis=1)
    
    # Drop rows with missing scores
    df.dropna(subset=['Total Score'], inplace=True)
    
    # Encode categorical features
    categorical_features = ['School', 'Director(s)', 'District', 'Level']
    if label_encoders is None:
        label_encoders = {col: LabelEncoder() for col in categorical_features}
    
    for col in categorical_features:
        if fit_encoders:
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        else:
            # Filter out rows with unseen labels for this column
            seen_labels = set(label_encoders[col].classes_)
            df = df[df[col].astype(str).isin(seen_labels)]
            df.loc[:, col] = label_encoders[col].transform(df[col].astype(str))
    
    return df, label_encoders

def train_and_predict(training_files, test_file):
    # Read and preprocess training data from multiple files
    training_data = []
    for file in training_files:
        df = pd.read_csv(file)
        training_data.append(df)
    
    # Combine training data into a single DataFrame
    combined_training_data = pd.concat(training_data)
    
    # Fit label encoders on the combined training data
    combined_training_data, label_encoders = preprocess_data(combined_training_data, fit_encoders=True)
    
    # Preprocess each training DataFrame with fitted label encoders
    preprocessed_training_data = []
    for df in training_data:
        df, _ = preprocess_data(df, label_encoders=label_encoders, fit_encoders=False)
        preprocessed_training_data.append(df)
    
    # Combine preprocessed training data into a single DataFrame
    combined_preprocessed_training_data = pd.concat(preprocessed_training_data)
    
    # Prepare feature matrix and target vector for training data
    features = ['School', 'Director(s)', 'District', 'Level']
    X_train = combined_preprocessed_training_data[features]
    y_train = combined_preprocessed_training_data['Total Score']
    
    # Train the Random Forest regressor on the training data
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    
    # Read and preprocess the test data
    df_test = pd.read_csv(test_file)
    df_test, _ = preprocess_data(df_test, label_encoders=label_encoders, fit_encoders=False)
    
    # Prepare feature matrix for test data
    X_test = df_test[features]
    
    # Make predictions on the test data
    y_pred = rf_regressor.predict(X_test)
    
    # Evaluate the model
    y_test = df_test['Total Score']
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    
    # Round predictions and evaluate again
    y_pred_rounded = y_pred.round()
    mse_rounded = mean_squared_error(y_test, y_pred_rounded)
    r2_rounded = r2_score(y_test, y_pred_rounded)
    print(f'Mean Squared Error (rounded): {mse_rounded}')
    print(f'R-squared (rounded): {r2_rounded}')
    
    # Calculate how many times the rounded predictions match the actual scores
    correct_predictions = (y_pred_rounded == y_test).sum()
    total_predictions = len(y_test)
    print(f'Correct Predictions: {correct_predictions} out of {total_predictions}')
    print(f'Accuracy: {correct_predictions / total_predictions:.2f}')
    
    # Create a DataFrame with the actual and predicted scores
    predictions = pd.DataFrame({
        'Index': df_test.index,
        'School': df_test['School'],
        'Actual Total Score': y_test,
        'Predicted Total Score': y_pred,
        'Predicted Total Score (Rounded)': y_pred_rounded
    })
    

    # ensure the School column is converted to integers explicitly
    predictions['School'] = predictions['School'].astype(int)

    # Decode the School names
    predictions['School'] = label_encoders['School'].inverse_transform(predictions['School'])
    
    # Print predictions for the test data
    print(predictions)

    # Specify the output file path
    output_file_path = 'predictions.csv'

    # Write the predictions DataFrame to a CSV file
    predictions.to_csv(output_file_path, index=False)
        
    # Print feature importances
    feature_importances = pd.DataFrame({
        'Feature': features,
        'Importance': rf_regressor.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(feature_importances)

# List of training files and test file
training_files = ['2018nc.csv', '2019nc.csv', '2022nc.csv']
test_file = '2023nc.csv'

# Train the model on the training files and predict the test file
train_and_predict(training_files, test_file)
