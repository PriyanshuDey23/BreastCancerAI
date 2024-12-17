import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_model(data): 
    """
    Create and train a Logistic Regression model with the given dataset.
    Scales the data and evaluates the model's performance.
    """
    logging.info("🚀 Preparing data for model training...")

    # Split data into features (X) and target (y)
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
  
    # Scale the data
    scaler = StandardScaler()  # It has to be on the same scale
    X = scaler.fit_transform(X)  # Not scaling y, because it is binary (0 or 1)
  
    # Split the data
    logging.info("🔄 Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
  
    # Train the model
    logging.info("🤖 Training the Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)
  
    # Test the model
    logging.info("📊 Evaluating the model's performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"✅ Model Accuracy: {accuracy:.2%}")
    logging.info("🔍 Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data(filepath="data/data.csv"):
    """
    Load and preprocess the dataset.
    Drops unnecessary columns, converts labels to binary, and handles missing data.
    """
    logging.info("📂 Loading and cleaning data...")
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"❌ File not found: {filepath}")
        raise
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    # Convert diagnosis labels to binary (M -> 1, B -> 0)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    logging.info("✅ Data loaded and cleaned successfully.")
    
    return data


def save_model(model, scaler, model_path="model/model.pkl", scaler_path="model/scaler.pkl"):
    """
    Save the trained model and scaler to disk using pickle.
    """
    logging.info("💾 Saving model and scaler...")
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"✅ Model saved at {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logging.info(f"✅ Scaler saved at {scaler_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save model or scaler: {e}")
        raise


def main():
    logging.info("🏁 Starting the model creation pipeline...")
    data = get_clean_data()  # Load and preprocess the data
    model, scaler = create_model(data)  # Train the model
    save_model(model, scaler)  # Save the model and scaler
    logging.info("🎉 Model pipeline completed successfully!")


if __name__ == '__main__':
    main()
