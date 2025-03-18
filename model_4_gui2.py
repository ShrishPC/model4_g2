import customtkinter as ctk
from tkinter import messagebox, Canvas
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained model (replace with your model path)
model = tf.keras.models.load_model('student_performance_model.h5')

# Define the columns and their possible values
numerical_cols = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                 'Tutoring_Sessions', 'Physical_Activity']
                 
ordinal_cols = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Far', 'Moderate', 'Near']
}

binary_cols = ['Extracurricular_Activities', 'Internet_Access', 
              'Learning_Disabilities', 'Gender']
              
nominal_cols = ['School_Type', 'Peer_Influence']

# Function to preprocess input data
def preprocess_input(input_data):
    # Create a copy to avoid modifying original dict
    processed = input_data.copy()
    
    for col, levels in ordinal_cols.items():
        processed[col] = levels.index(processed[col])
    
    processed['Extracurricular_Activities'] = 1 if processed['Extracurricular_Activities'] == 'Yes' else 0
    processed['Internet_Access'] = 1 if processed['Internet_Access'] == 'Yes' else 0
    processed['Learning_Disabilities'] = 1 if processed['Learning_Disabilities'] == 'Yes' else 0
    processed['Gender'] = 1 if processed['Gender'] == 'Male' else 0
    
    # One-hot encoding for nominal columns
    processed['School_Type_Public'] = 1 if processed['School_Type'] == 'Public' else 0
    processed['School_Type_Private'] = 1 if processed['School_Type'] == 'Private' else 0
    
    peer_influence = processed['Peer_Influence']
    processed['Peer_Influence_Positive'] = 1 if peer_influence == 'Positive' else 0
    processed['Peer_Influence_Negative'] = 1 if peer_influence == 'Negative' else 0
    processed['Peer_Influence_Neutral'] = 1 if peer_influence == 'Neutral' else 0
    
    del processed['School_Type']
    del processed['Peer_Influence']
    
    # Convert to DataFrame and scale
    input_df = pd.DataFrame([processed])
    scaler = StandardScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    return input_df

# Function to predict the score
def predict_score():
    try:
        input_data = {}
        
        # Get slider values
        for col in numerical_cols:
            input_data[col] = float(sliders[col].get())
            
        # Ordinal inputs
        for col in ordinal_cols:
            input_data[col] = entries[col].get()
            
        # Binary inputs
        for col in binary_cols:
            input_data[col] = entries[col].get()
            
        # Nominal inputs
        for col in nominal_cols:
            input_data[col] = entries[col].get()
            
        # Process and predict
        input_df = preprocess_input(input_data)
        prediction = model.predict(input_df)
        messagebox.showinfo("Prediction", f"Predicted Exam Score: {prediction[0][0]:.2f}")
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("Student Exam Score Predictor")
app.geometry("800x600")
app.resizable(True, True)

# Configure grid layout
app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(0, weight=1)

# Create main container
canvas = Canvas(app, bg="#2b2b2b", highlightthickness=0)
scrollbar = ctk.CTkScrollbar(app, orientation='vertical', command=canvas.yview)
scrollable_frame = ctk.CTkFrame(canvas, fg_color="#2b2b2b")

scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Styling parameters
label_font = ("Arial", 12, "bold")
entry_font = ("Arial", 12)
section_font = ("Arial", 14, "bold")
button_font = ("Arial", 14, "bold")
entry_width = 300

sliders = {}
entries = {}

# Create slider function
def create_slider(parent, label_text, min_val, max_val):
    frame = ctk.CTkFrame(parent, fg_color="transparent")
    frame.pack(fill="x", padx=20, pady=(0, 10))
    
    # Label and value display
    ctk.CTkLabel(frame, text=label_text, font=label_font, anchor="w").pack(side="left", padx=(0, 10))
    value_label = ctk.CTkLabel(frame, text="0", font=entry_font)
    value_label.pack(side="right")
    
    # Slider
    slider = ctk.CTkSlider(
        frame,
        from_=min_val,
        to=max_val,
        number_of_steps=int(max_val - min_val),
        command=lambda val: value_label.configure(text=f"{float(val):.1f}")
    )
    slider.pack(fill="x", expand=True)
    slider.set(min_val)
    
    return slider

# Numerical Inputs Section with Sliders
slider_ranges = {
    'Hours_Studied': (0, 24),
    'Attendance': (0, 100),
    'Sleep_Hours': (0, 12),
    'Previous_Scores': (0, 100),
    'Tutoring_Sessions': (0, 10),
    'Physical_Activity': (0, 7)
}

for col in numerical_cols:
    min_val, max_val = slider_ranges[col]
    sliders[col] = create_slider(scrollable_frame, col, min_val, max_val)

# Other input sections
for col, levels in ordinal_cols.items():
    ctk.CTkLabel(scrollable_frame, text=col, text_color="white", font=label_font).pack(pady=5)
    entries[col] = ctk.CTkComboBox(scrollable_frame, values=levels, font=entry_font, width=entry_width)
    entries[col].pack(pady=5)

for col in binary_cols:
    ctk.CTkLabel(scrollable_frame, text=col, text_color="white", font=label_font).pack(pady=5)
    entries[col] = ctk.CTkComboBox(scrollable_frame, values=['Yes', 'No'], font=entry_font, width=entry_width)
    entries[col].pack(pady=5)

for col in nominal_cols:
    ctk.CTkLabel(scrollable_frame, text=col, text_color="white", font=label_font).pack(pady=5)
    if col == 'School_Type':
        entries[col] = ctk.CTkComboBox(scrollable_frame, values=['Public', 'Private'], font=entry_font, width=entry_width)
    elif col == 'Peer_Influence':
        entries[col] = ctk.CTkComboBox(scrollable_frame, values=['Positive', 'Negative', 'Neutral'], font=entry_font, width=entry_width)
    entries[col].pack(pady=5)

# Predict Button
ctk.CTkButton(
    scrollable_frame, 
    text="Predict Score", 
    command=predict_score,
    font=button_font,
    fg_color="#00bcd4",
    hover_color="#0097a7",
    height=40
).pack(pady=20)

canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
scrollbar.pack(side="right", fill="y")

app.mainloop()