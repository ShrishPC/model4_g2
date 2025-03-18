import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf

# Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# Define columns
numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
ordinal_cols = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Far', 'Moderate', 'Near']
}
binary_cols = ['Extracurricular_Activities', 'Internet_Access', 'Learning_Disabilities', 'Gender']
nominal_cols = ['School_Type', 'Peer_Influence']

# Handle missing values
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
for col in ordinal_cols.keys():
    df[col] = df[col].fillna(df[col].mode()[0])
for col in binary_cols + nominal_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode ordinal columns
for col, levels in ordinal_cols.items():
    df[col] = df[col].map({level: i for i, level in enumerate(levels)})

# Encode binary columns
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
df['Internet_Access'] = df['Internet_Access'].map({'Yes': 1, 'No': 0})
df['Learning_Disabilities'] = df['Learning_Disabilities'].map({'Yes': 1, 'No': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# One-hot encode nominal columns
df = pd.get_dummies(df, columns=nominal_cols)

# Separate features and target
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Add early stopping to prevent overfitting
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop]
)

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {test_mae:.2f}")

model.save('student_performance_model.h5')
print("successfully save")