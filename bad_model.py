import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('2023nc.csv')

roman_to_int = {
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4,
    'V': 5,
    'VI': 6,
    'MW': 7,
    'I/II': 1.5,
    'II/III': 2.5,
    'III/IV': 3.5,
    'IV/V': 4.5,
    'V/VI': 5.5,
    'VI/MW': 6.5,  
    'co': None,  
    'C/O': None, 
}

df['Judge 1'] = df['Judge 1'].map(roman_to_int)
df['Judge 2'] = df['Judge 2'].map(roman_to_int)
df['Judge 3'] = df['Judge 3'].map(roman_to_int)

df['Total Score'] = df[['Judge 1', 'Judge 2', 'Judge 3']].sum(axis=1)

df.dropna(subset=['Total Score'], inplace=True)

categorical_features = ['School', 'Director(s)', 'District', 'Level']
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = ['School', 'Director(s)', 'District', 'Level']
X = df[features]
y = df['Total Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

feature_importances = rf_regressor.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f'{feature}: {importance}')


