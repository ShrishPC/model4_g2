import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tkinter as tk
from tkinter import messagebox, ttk

# Load the dataset
df = pd.read_csv("StudentPerformanceFactors.csv")  # Update with actual file name

# Define categorical and numerical columns
categorical_columns = [
    'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
    'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'School_Type',
    'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 'Distance_from_Home', 'Gender'
]
numerical_columns = [col for col in df.columns if col not in categorical_columns + ['Exam_score']]

# Encode categorical features
label_encoders = {}
categorical_options = {}  # Store valid options for each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    categorical_options[col] = list(le.classes_)  # Store valid categories

# Selecting features and target
feature_columns = categorical_columns + numerical_columns
X = df[feature_columns]
y = df[['Exam_Score']]  # Predicting exam score

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")

# GUI for user input
def predict_score():
    try:
        user_data = []
        for col, widget in zip(feature_columns, input_widgets):
            value = widget.get()
            if col in categorical_columns:
                if value in categorical_options[col]:
                    value = label_encoders[col].transform([value])[0]  # Encode categorical inputs
                else:
                    messagebox.showerror("Error", f"Invalid input for {col}. Choose from: {', '.join(categorical_options[col])}")
                    return
            else:
                value = float(value)  # Convert numerical inputs
            user_data.append(value)
        
        user_data = np.array(user_data).reshape(1, -1)
        user_data = scaler.transform(user_data)
        prediction = model.predict(user_data)
        messagebox.showinfo("Prediction", f"Predicted Exam Score: {prediction[0][0]:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create GUI
root = tk.Tk()
root.title("Exam Score Predictor")

tk.Label(root, text="Enter Student Details Below:", font=("Arial", 14, "bold")).pack()

input_widgets = []
for col in feature_columns:
    frame = tk.Frame(root)
    frame.pack()
    tk.Label(frame, text=col + ":").pack(side=tk.LEFT)
    
    if col in categorical_columns:
        combo = ttk.Combobox(frame, values=categorical_options[col])
        combo.pack(side=tk.RIGHT)
        input_widgets.append(combo)
    else:
        entry = tk.Entry(frame)
        entry.pack(side=tk.RIGHT)
        input_widgets.append(entry)

tk.Button(root, text="Predict Score", command=predict_score).pack()

root.mainloop()
