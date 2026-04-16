import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_preprocessing import load_and_preprocess_data


def main():
    file_path = "../data/creditcard.csv"

    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)

    # Create model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "fraud_model.pkl")
    print("\nModel saved as fraud_model.pkl")


if __name__ == "__main__":
    main()