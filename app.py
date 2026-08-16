import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# Title
st.title("🌱 Dry Bean Classification")
st.write(
    "Machine Learning application for classifying dry bean varieties "
    "using morphological features."
)

st.divider()

# Model directory
MODEL_DIR = "models"

# Load models
logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

decision_tree_model = joblib.load(
    os.path.join(MODEL_DIR, "decision_tree.pkl")
)

knn_model = joblib.load(
    os.path.join(MODEL_DIR, "knn.pkl")
)

naive_bayes_model = joblib.load(
    os.path.join(MODEL_DIR, "naive_bayes.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

# Load preprocessing objects
scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

feature_names = joblib.load(
    os.path.join(MODEL_DIR, "feature_names.pkl")
)

# Model selection
st.subheader("Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Gaussian Naive Bayes",
        "Random Forest"
    ]
)

if model_name == "Logistic Regression":
    model = logistic_model
    use_scaler = True

elif model_name == "Decision Tree":
    model = decision_tree_model
    use_scaler = False

elif model_name == "KNN":
    model = knn_model
    use_scaler = True

elif model_name == "Gaussian Naive Bayes":
    model = naive_bayes_model
    use_scaler = True

else:
    model = random_forest_model
    use_scaler = False

st.info(f"Selected model: **{model_name}**")

st.divider()

# Upload test data
st.subheader("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 16 morphological features",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    st.success(
        f"File uploaded successfully: {uploaded_file.name}"
    )

    st.write(
        "The uploaded file contains the data used for prediction."
    )

    st.dataframe(test_data)

else:

    st.info(
        "Please upload a CSV file to continue."
    )

# Prediction
if uploaded_file is not None and st.button("🔍 Predict"):

    input_data = test_data[feature_names].copy()

    if use_scaler:
        input_for_model = scaler.transform(input_data)
    else:
        input_for_model = input_data

    # Generate predictions
    predictions = model.predict(input_for_model)

    predicted_classes = label_encoder.inverse_transform(
        predictions.astype(int)
    )

    # Generate probabilities
    probabilities = model.predict_proba(input_for_model)

    max_probabilities = np.max(probabilities, axis=1)

    # Results
    st.subheader("Prediction Results")

    results = input_data.copy()

    results["Predicted_Class"] = predicted_classes
    results["Prediction_Probability"] = max_probabilities

    if "Actual_Class" in test_data.columns:

        results["Actual_Class"] = test_data["Actual_Class"]

        results["Correct"] = (
            results["Predicted_Class"]
            == results["Actual_Class"]
        )

    st.dataframe(results)

    # Summary
    st.subheader("Prediction Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model", model_name)

    with col2:
        st.metric("Number of Samples", len(predictions))

    with col3:
        st.metric(
            "Average Probability",
            f"{max_probabilities.mean():.2%}"
        )
    # Evaluation and comparison
    if "Actual_Class" in test_data.columns:

        actual_classes = test_data["Actual_Class"].values

        # Evaluation Metrics
        accuracy = accuracy_score(
            actual_classes,
            predicted_classes
        )

        precision = precision_score(
            actual_classes,
            predicted_classes,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            actual_classes,
            predicted_classes,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            actual_classes,
            predicted_classes,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(
            actual_classes,
            predicted_classes
        )

        auc = roc_auc_score(
            actual_classes,
            probabilities,
            multi_class="ovr",
            average="weighted"
        )

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")

        with col2:
            st.metric("AUC", f"{auc:.4f}")
            st.metric("Recall", f"{recall:.4f}")

        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC", f"{mcc:.4f}")

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")

        cm = confusion_matrix(
            actual_classes,
            predicted_classes,
            labels=label_encoder.classes_
        )

        cm_df = pd.DataFrame(
            cm,
            index=label_encoder.classes_,
            columns=label_encoder.classes_
        )

        st.dataframe(cm_df)

        # Classification Report
        st.subheader("📋 Classification Report")

        report = classification_report(
            actual_classes,
            predicted_classes,
            labels=label_encoder.classes_,
            target_names=label_encoder.classes_,
            zero_division=0
        )

        st.code(report)

        # Compare All Models
        st.divider()
        st.subheader("🏆 Model Comparison")

        models = {
            "Logistic Regression": (logistic_model, True),
            "Decision Tree": (decision_tree_model, False),
            "KNN": (knn_model, True),
            "Gaussian Naive Bayes": (naive_bayes_model, True),
            "Random Forest": (random_forest_model, False)
        }

        comparison_results = []

        for name, (current_model, scale_data) in models.items():

            if scale_data:
                comparison_input = scaler.transform(
                    test_data[feature_names]
                )
            else:
                comparison_input = test_data[feature_names]

            model_predictions = current_model.predict(
                comparison_input
            )

            model_probabilities = current_model.predict_proba(
                comparison_input
            )

            predicted_labels = label_encoder.inverse_transform(
                model_predictions.astype(int)
            )

            model_accuracy = accuracy_score(
                actual_classes,
                predicted_labels
            )

            model_precision = precision_score(
                actual_classes,
                predicted_labels,
                average="weighted",
                zero_division=0
            )

            model_recall = recall_score(
                actual_classes,
                predicted_labels,
                average="weighted",
                zero_division=0
            )

            model_f1 = f1_score(
                actual_classes,
                predicted_labels,
                average="weighted",
                zero_division=0
            )

            model_mcc = matthews_corrcoef(
                actual_classes,
                predicted_labels
            )

            model_auc = roc_auc_score(
                actual_classes,
                model_probabilities,
                multi_class="ovr",
                average="weighted"
            )

            comparison_results.append({
                "ML Model Name": name,
                "Accuracy": model_accuracy,
                "AUC": model_auc,
                "Precision": model_precision,
                "Recall": model_recall,
                "F1": model_f1,
                "MCC": model_mcc
            })

        comparison_df = pd.DataFrame(
            comparison_results
        )

        st.dataframe(
            comparison_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )
 