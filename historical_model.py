import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

def standardize_school_name(school_name):
    priority_words = ['symphonic', 'wind', 'concert', '7th', '8th', 'middle', 'high']
    parts = school_name.lower().split()
    
    # extract the first word
    first_word = parts[0]
    
    # extract the second word if it's not a priority word
    second_word = parts[1] if len(parts) > 1 and parts[1] not in priority_words else ''
    
    # find the highest priority word in the name
    highest_priority_word = ''
    for word in priority_words:
        if word in parts:
            highest_priority_word = word
            break
    
    # create the standardized name
    standardized_name = first_word
    if second_word:
        standardized_name += f" {second_word}"
    if highest_priority_word:
        standardized_name += f" {highest_priority_word.capitalize()}"
    
    return standardized_name

def preprocess_data(df, label_encoders=None, fit_encoders=False):
    # define a one-to-one mapping from roman numerals to integers
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
        'co': None,  # comments only
        'C/O': None  # comments only
    }
    
    # standardize school names
    df['School'] = df['School'].apply(standardize_school_name)
    
    # replace Roman numerals with integers
    df['Judge 1'] = df['Judge 1'].map(roman_to_int)
    df['Judge 2'] = df['Judge 2'].map(roman_to_int)
    df['Judge 3'] = df['Judge 3'].map(roman_to_int)
    
    # calculate the total score across all judges, lower is better
    df['Total Score'] = df[['Judge 1', 'Judge 2', 'Judge 3']].sum(axis=1)
    
    # drop rows with missing scores
    df.dropna(subset=['Total Score'], inplace=True)
    
    # encode categorical features
    categorical_features = ['School', 'Director(s)', 'District', 'Level']
    if label_encoders is None:
        label_encoders = {col: LabelEncoder() for col in categorical_features}
    
    for col in categorical_features:
        if fit_encoders:
            df[col] = label_encoders[col].fit_transform(df[col])
        else:
            # filter out rows with unseen labels for this column
            seen_labels = set(label_encoders[col].classes_)
            df = df[df[col].isin(seen_labels)]
            df.loc[:, col] = label_encoders[col].transform(df[col])
    
    return df, label_encoders

# read and preprocess the 2022 data
df_2022 = pd.read_csv('2022nc.csv')
df_2022, label_encoders = preprocess_data(df_2022, fit_encoders=True)

# prepare feature matrix and target vector for 2022 data
features = ['School', 'Director(s)', 'District', 'Level']
X_train = df_2022[features]
y_train = df_2022['Total Score']

# train the Random Forest regressor on the 2022 data
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# read and preprocess the 2023 data
df_2023 = pd.read_csv('2023nc.csv')
df_2023, _ = preprocess_data(df_2023, label_encoders=label_encoders, fit_encoders=False)

# prepare feature matrix for 2023 data
X_test = df_2023[features]

# make predictions on the 2023 data
y_pred = rf_regressor.predict(X_test)

# evaluate the model
y_test = df_2023['Total Score']
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# round predictions and evaluate again
y_pred_rounded = y_pred.round()
mse_rounded = mean_squared_error(y_test, y_pred_rounded)
r2_rounded = r2_score(y_test, y_pred_rounded)
print(f'Mean Squared Error (rounded): {mse_rounded}')
print(f'R-squared (rounded): {r2_rounded}')

# create a dateframe with the actual and predicted scores
predictions_2023 = pd.DataFrame({
    'Index': df_2023.index,
    'School': df_2023['School'],
    'Actual Total Score': y_test,
    'Predicted Total Score': y_pred,
    'Predicted Total Score (Rounded)': y_pred_rounded
})

# ensure the School column is converted to integers explicitly
predictions_2023['School'] = predictions_2023['School'].astype(int)

# decode the School names
predictions_2023['School'] = label_encoders['School'].inverse_transform(predictions_2023['School'])

# print predictions for the 2023 data
print(predictions_2023)

# accuracy rounded to the nearest integer
correct_predictions = (y_pred_rounded == y_test).sum()
total_predictions = len(y_test)
print(f'Correct Predictions: {correct_predictions} out of {total_predictions}')
print(f'Accuracy: {correct_predictions / total_predictions:.2f}')

# print feature importances
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_regressor.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(feature_importances)
