import math
import streamlit as st
import pandas as pd
from pipeline.config import IMPUTATION_OPTIONS, SCALING_OPTIONS, MODEL_OPTIONS, DATASET_OPTIONS
from pipeline.data_loader import load_pima, load_frankfurt, load_custom_csv
from pipeline.preprocess import get_imputer, get_scaler, fit_preprocessing, transform_input
from pipeline.models import get_model, PARAM_GRIDS, train_and_predict, tune_model
    
# Inject custom CSS to style the "Predict Risk" button
st.markdown(
    """
    <style>
    /* Button default style */
    .stButton>button {
        border-color: #4CAF50;
        border-radius: 10px;
        padding: 0.6em 1.2em;
    }
    /* Button hover style */
    .stButton>button:hover {
        background-color: #5CAF50;
        color: #fff;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Risk Predictor")

st.info("""
**Note:** By default, the model is trained on all available data. This is suitable for demonstration but not for real-world deployment. Optionally, you can enable a train/test split below to see how the model performs on unseen data.
""")

use_split = st.sidebar.checkbox("Use train/test split for evaluation (demo)")
test_size = st.sidebar.slider("Test size (fraction)", min_value=0.1, max_value=0.5, value=0.3, step=0.05) if use_split else None

# Caching data loading
@st.cache_data(show_spinner=False)
def cached_load_data(dataset_choice, uploaded=None):
    if dataset_choice == 'pima':
        return load_pima()
    elif dataset_choice == 'frankfurt':
        return load_frankfurt()
    else:
        if uploaded:
            return load_custom_csv(uploaded)
        else:
            return None, None

# Caching preprocessing
@st.cache_resource(show_spinner=False)
def cached_preprocessing(X, imputer_choice, scaler_choice):
    imputer = get_imputer(imputer_choice)
    scaler = get_scaler(scaler_choice)
    X_proc, imputer, scaler = fit_preprocessing(X, imputer, scaler)
    return X_proc, imputer, scaler

# Caching model training
@st.cache_resource(show_spinner=False)
def cached_train_model(X_proc, y, model_choice, do_tune):
    if do_tune:
        base_model = get_model(model_choice)
        best_model, best_params, best_score = tune_model(base_model, PARAM_GRIDS[model_choice], X_proc, y)
        return best_model, best_params, best_score
    else:
        model = get_model(model_choice)
        model.fit(X_proc, y)
        return model, None, None

def run_training_and_predict(X_proc, y, input_proc, model_choice, do_tune, use_split, test_size, return_test=False):
    """Unified function to handle model training, tuning, prediction, and optional test split. Optionally returns test set results."""
    from sklearn.metrics import accuracy_score
    import numpy as np
    if use_split:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=42)
        model, best_params, best_score = cached_train_model(X_train, y_train, model_choice, do_tune)
        preds, probs = train_and_predict(model, X_train, y_train, input_proc)
        test_preds, test_probs = train_and_predict(model, X_train, y_train, X_test)
        test_acc = accuracy_score(y_test, test_preds)
        if hasattr(model, "predict_proba"):
            test_probs_out = test_probs[:, 1] if test_probs is not None and len(test_probs.shape) > 1 else test_probs
        else:
            test_probs_out = test_probs
        if return_test:
            return preds, probs, best_params, test_acc, y_test, test_preds, test_probs_out
        else:
            return preds, probs, best_params, test_acc
    else:
        model, best_params, best_score = cached_train_model(X_proc, y, model_choice, do_tune)
        preds, probs = train_and_predict(model, X_proc, y, input_proc)
        if return_test:
            return preds, probs, best_params, None, None, None, None
        else:
            return preds, probs, best_params, None

# Sidebar controls
dataset_choice = st.sidebar.selectbox("Choose dataset", DATASET_OPTIONS)
imputer_choice = st.sidebar.selectbox("Imputation method", IMPUTATION_OPTIONS)
scaler_choice = st.sidebar.selectbox("Scaling method", SCALING_OPTIONS)
model_choice = st.sidebar.selectbox("Model", MODEL_OPTIONS)
do_tune = st.sidebar.checkbox("Use hyperparameter tuning")

# Load data
X, y = None, None
try:
    if dataset_choice == 'custom':
        uploaded = st.sidebar.file_uploader("Upload CSV with 'Outcome' column", type='csv')
        if uploaded:
            X, y = cached_load_data(dataset_choice, uploaded)
        else:
            st.warning("Please upload a valid CSV file.")
            st.stop()
    else:
        X, y = cached_load_data(dataset_choice)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Field descriptions
FIELD_HELP = {
    'Pregnancies': 'Number of times pregnant (integer)',
    'Glucose': 'Plasma glucose concentration (mg/dL)',
    'BloodPressure': 'Diastolic blood pressure (mm Hg)',
    'SkinThickness': 'Triceps skin fold thickness (mm)',
    'Insulin': '2-Hour serum insulin (µU/mL)',
    'BMI': 'Body mass index (weight in kg/(height in m)^2)',
    'DiabetesPedigreeFunction': '''
    The Diabetes Pedigree Function (DPF) is not something you “measure” directly with a device—it’s a composite score derived from your family history of diabetes and how closely related those relatives are, adjusted for your age. In the original Pima dataset it’s computed as:
    DPF = (Σ (relation_weight × diabetes_status)) / age
    where “relation_coefficient” encodes how genetically close each relative is (e.g. parent = 0.5, sibling = 0.5, grandparent = 0.25), “diabetes_status” is 1 if they have diabetes, and you also factor in ages.
    ''',
    'Age': 'Age of the patient (years)'
}

# Feature input form
if X is not None:
    st.header("Enter feature values:")
    input_data = {}
    for col in X.columns:
        help_text = FIELD_HELP.get(col, '')
        if col in ['Pregnancies', 'Glucose', 'BloodPressure', 'Age']:
            input_data[col] = int(
                st.number_input(
                    f"{col}", value=int(X[col].mean()), help=help_text
                )
            )
        else:
            input_data[col] = st.number_input(
                f"{col}", value=float(X[col].mean()), help=help_text
            )
    input_df = pd.DataFrame([input_data])
else:
    input_df = None

# On-click: process & predict
if st.sidebar.button("Prediction"):
    if X is None or input_df is None:
        st.error("Data not loaded or input not ready.")
        st.stop()
    try:
        with st.spinner("Preprocessing data..."):
            X_proc, imputer, scaler = cached_preprocessing(X, imputer_choice, scaler_choice)
            input_proc = transform_input(input_df, imputer, scaler, X.columns)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    # st.write("Processed input:", input_proc)
    try:
        with st.spinner("Training and predicting..."):
            preds, probs, best_params, test_acc, test_y, test_preds, test_probs = run_training_and_predict(
                X_proc, y, input_proc, model_choice, do_tune, use_split, test_size if use_split else None, return_test=True
            )
        if test_acc is not None:
            st.success(f"Test set accuracy: {test_acc:.3f}")
            # Show confusion matrix and ROC curve
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns
            cm = confusion_matrix(test_y, test_preds)
            st.subheader("Confusion Matrix (Test Set)")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            # ROC curve
            if test_probs is not None:
                fpr, tpr, _ = roc_curve(test_y, test_probs)
                roc_auc = auc(fpr, tpr)
                st.subheader("ROC Curve (Test Set)")
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("Receiver Operating Characteristic")
                ax2.legend(loc="lower right")
                st.pyplot(fig2)
    except Exception as e:
        st.error(f"Error during model training or prediction: {e}")
        st.stop()

    if do_tune and best_params is not None:
        st.write("Best Params:", best_params)
        # st.write("CV Accuracy:", f"{best_score:.4f}")

    result = {
        0: "Non Diabetic",
        1: "Diabetic"
    }

    st.sidebar.subheader("Prediction Result")
    try:
        if preds is not None and len(preds) > 0:
            st.sidebar.write("**Class:**", result.get(int(preds[0]), "Unknown"))
            if probs is not None:
                st.sidebar.write("**Risk Probability:**", f"{(probs[0] * 100):.2f} %")
        else:
            st.sidebar.write("Prediction could not be displayed due to an error.")
    except Exception as e:
        st.sidebar.write("Prediction could not be displayed due to an error.")
        