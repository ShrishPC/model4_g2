import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def load_and_preprocess_data(filepath, target_column):
    """
    Load and preprocess the dataset.
    
    Args:
        filepath (str): Path to the dataset file.
        target_column (str): Name of the target column.
    
    Returns:
        X_train, X_test, y_train, y_test: Split and preprocessed data.
        label_encoders (dict): Dictionary of label encoders for categorical columns.
        scaler (StandardScaler): Fitted scaler for feature standardization.
    """
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df = df.dropna()
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Split the data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler

def build_model(input_dim):
    """
    Build a TensorFlow model.
    
    Args:
        input_dim (int): Number of input features.
    
    Returns:
        model: Compiled TensorFlow model.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    
    # Hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    #model.add(Dense(16, activation='relu'))
    
    # Output layer
    model.add(Dense(1))  # Use 'sigmoid' for binary classification
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Use 'binary_crossentropy' for binary classification
    
    return model

def train_model(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """
    Train the TensorFlow model.
    
    Args:
        model: TensorFlow model.
        X_train: Training features.
        y_train: Training target.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        validation_split (float): Fraction of training data to use for validation.
    
    Returns:
        history: Training history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the TensorFlow model.
    
    Args:
        model: TensorFlow model.
        X_test: Testing features.
        y_test: Testing target.
    
    Returns:
        loss, mae: Loss and mean absolute error on the test data.
    """
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Mean Absolute Error on Test Data: {mae}')
    return loss, mae

def make_predictions(model, X_test):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained TensorFlow model.
        X_test: Testing features.
    
    Returns:
        predictions: Predicted values.
    """
    predictions = model.predict(X_test)
    return predictions

def save_model(model, filepath):
    """
    Save the trained model to a file.
    
    Args:
        model: Trained TensorFlow model.
        filepath (str): Path to save the model.
    """
    model.save(filepath)

def load_saved_model(filepath):
    """
    Load a saved model from a file.
    
    Args:
        filepath (str): Path to the saved model.
    
    Returns:
        model: Loaded TensorFlow model.
    """
    from tensorflow.keras.models import load_model
    model = load_model(filepath)
    return model

# Example usage
if __name__ == "__main__":
    # Load and preprocess the data
    filepath = "StudentPerformanceFactors.csv"
    target_column = "Exam_Score"  # Replace with the actual target column name
    X_train, X_test, y_train, y_test, label_encoders, scaler = load_and_preprocess_data(filepath, target_column)
    
    # Build the model
    model = build_model(X_train.shape[1])
    
    # Train the model
    history = train_model(model, X_train, y_train)
    
    # Evaluate the model
    loss, mae = evaluate_model(model, X_test, y_test)
    
    # Make predictions
    predictions = make_predictions(model, X_test)
    print(predictions[:5])
    
    # Save the model (optional)
    #save_model(model, 'student_performance_model.h5')
    
    # Load the model (optional)
    #loaded_model = load_saved_model('student_performance_model.h5')