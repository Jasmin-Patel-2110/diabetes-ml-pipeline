import math
import streamlit as st
import pandas as pd
from pipeline.config import IMPUTATION_OPTIONS, SCALING_OPTIONS, MODEL_OPTIONS, DATASET_OPTIONS
from pipeline.data_loader import load_pima, load_frankfurt, load_custom_csv
from pipeline.preprocess import get_imputer, get_scaler, fit_preprocessing, transform_input
from pipeline.models import get_model, PARAM_GRIDS, train_and_predict, tune_model

# --- Page Config ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stButton>button {
        border-color: #4CAF50;
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Caching Functions ---
@st.cache_data(show_spinner="Loading data...")
def cached_load_data(dataset_choice, uploaded=None):
    if dataset_choice == 'pima':
        return load_pima()
    elif dataset_choice == 'frankfurt':
        return load_frankfurt()
    elif uploaded:
        return load_custom_csv(uploaded)
    return None, None

@st.cache_resource(show_spinner="Fitting preprocessors...")
def cached_preprocessing(_X, imputer_choice, scaler_choice):
    imputer = get_imputer(imputer_choice)
    scaler = get_scaler(scaler_choice)
    X_proc, imputer, scaler = fit_preprocessing(_X, imputer, scaler)
    return X_proc, imputer, scaler

@st.cache_resource(show_spinner="Training model...")
def cached_train_model(_X_proc, _y, model_choice, do_tune):
    if do_tune:
        base_model = get_model(model_choice)
        best_model, best_params, best_score = tune_model(base_model, PARAM_GRIDS[model_choice], _X_proc, _y)
        return best_model, best_params, best_score
    else:
        model = get_model(model_choice)
        model.fit(_X_proc, _y)
        return model, None, None

def run_training_and_predict(X_proc, y, input_proc, model_choice, do_tune, use_split, test_size):
    from sklearn.metrics import accuracy_score
    import numpy as np

    if use_split:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=test_size, random_state=42)
        model, best_params, _ = cached_train_model(X_train, y_train, model_choice, do_tune)
        preds, probs = train_and_predict(model, X_train, y_train, input_proc)
        test_preds, test_probs = train_and_predict(model, X_train, y_train, X_test)
        test_acc = accuracy_score(y_test, test_preds)
        
        test_probs_out = test_probs[:, 1] if hasattr(model, "predict_proba") and test_probs is not None and len(test_probs.shape) > 1 else test_probs
        return preds, probs, best_params, test_acc, y_test, test_preds, test_probs_out
    else:
        model, best_params, _ = cached_train_model(X_proc, y, model_choice, do_tune)
        preds, probs = train_and_predict(model, X_proc, y, input_proc)
        return preds, probs, best_params, None, None, None, None

# --- Main App ---
st.title("🩺 Diabetes Risk Predictor")

# Initialize session state for prediction results
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.result = {}

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["**Predict Your Risk**", "**Model Performance & Settings**", "**About**"])

# --- Tab 2: Model Performance & Settings ---
with tab2:
    st.header("Model Configuration")
    st.info("Adjust the settings below to change how the model is trained. These settings directly impact performance.")

    c1, c2 = st.columns(2)
    with c1:
        dataset_choice = st.selectbox("Choose a Dataset", DATASET_OPTIONS)
        model_choice = st.selectbox("Choose a Model", MODEL_OPTIONS)
        imputer_choice = st.selectbox("Imputation Method", IMPUTATION_OPTIONS)
        scaler_choice = st.selectbox("Scaling Method", SCALING_OPTIONS)
    with c2:
        do_tune = st.checkbox("Use Hyperparameter Tuning", help="Find the best model parameters. Can be slower.")
        use_split = st.checkbox("Use Train/Test Split", value=True, help="Evaluate the model on unseen data.")
        test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.3, step=0.05) if use_split else None

    # Performance display area
    st.header("Model Performance")
    if not st.session_state.prediction_made:
        st.warning("Make a prediction to see performance metrics here.")
    else:
        if st.session_state.result.get('test_acc') is not None:
            st.success(f"**Test Set Accuracy:** {st.session_state.result['test_acc']:.3f}")
            
            c1, c2 = st.columns(2)
            with c1:
                # Confusion Matrix
                from sklearn.metrics import confusion_matrix
                import seaborn as sns
                import matplotlib.pyplot as plt
                cm = confusion_matrix(st.session_state.result['y_test'], st.session_state.result['test_preds'])
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            with c2:
                # ROC Curve
                if st.session_state.result['test_probs'] is not None:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(st.session_state.result['y_test'], st.session_state.result['test_probs'])
                    roc_auc = auc(fpr, tpr)
                    st.subheader("ROC Curve")
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.legend(loc="lower right")
                    st.pyplot(fig2)
        else:
            st.info("Train/Test split was not used, so no out-of-sample performance metrics are available.")
        
        if st.session_state.result.get('best_params'):
            st.write("**Best Hyperparameters Found:**", st.session_state.result['best_params'])

# --- Data Loading (depends on settings in Tab 2) ---
X, y = None, None
if dataset_choice == 'custom':
    uploaded = st.file_uploader("Upload your CSV", type='csv', help="Must contain an 'Outcome' column.")
    if uploaded:
        X, y = cached_load_data(dataset_choice, uploaded)
else:
    X, y = cached_load_data(dataset_choice)

# --- Sidebar for User Input ---
with st.sidebar:
    st.header("Enter Patient Data")
    if X is None:
        st.warning("Please select or upload a dataset first.")
        st.stop()
    
    input_data = {}
    for col in X.columns:
        if col in ['Pregnancies', 'Glucose', 'BloodPressure', 'Age']:
            input_data[col] = st.number_input(f"Enter {col}", value=int(X[col].mean()), min_value=0, max_value=300)
        else:
            input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()), min_value=0.0, format="%.2f")
    input_df = pd.DataFrame([input_data])

    if st.button("Get Prediction"):
        try:
            X_proc, imputer, scaler = cached_preprocessing(X, imputer_choice, scaler_choice)
            input_proc = transform_input(input_df, imputer, scaler, X.columns)
            
            preds, probs, best_params, test_acc, y_test, test_preds, test_probs = run_training_and_predict(
                X_proc, y, input_proc, model_choice, do_tune, use_split, test_size
            )
            
            # Store results in session state
            st.session_state.prediction_made = True
            st.session_state.result = {
                'prediction': preds[0] if preds is not None else None,
                'probability': probs[0] * 100 if probs is not None else None,
                'best_params': best_params,
                'test_acc': test_acc,
                'y_test': y_test,
                'test_preds': test_preds,
                'test_probs': test_probs
            }

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- Tab 1: Prediction Result ---
with tab1:
    st.header("Your Diabetes Risk Assessment")
    if not st.session_state.prediction_made:
        st.info("Enter your data in the sidebar and click 'Get Prediction' to see your result.")
    else:
        pred_val = st.session_state.result.get('prediction')
        prob_val = st.session_state.result.get('probability')

        if pred_val == 1:
            st.metric(label="Risk Assessment", value="Diabetic", delta=f"{prob_val:.2f}% Probability", delta_color="inverse")
            st.error("Based on the data provided, the model predicts a high risk of diabetes. Please consult a healthcare professional for advice.", icon="⚠️")
        elif pred_val == 0:
            st.metric(label="Risk Assessment", value="Non-Diabetic", delta=f"{prob_val:.2f}% Probability", delta_color="normal")
            st.success("Based on the data provided, the model predicts a low risk of diabetes. Continue to maintain a healthy lifestyle.", icon="✅")
        else:
            st.warning("Could not make a prediction. Please check your inputs or settings.")

# --- Tab 3: About ---
with tab3:
    st.header("About This Project")
    st.markdown("""
    This interactive tool is the practical implementation of our research on improving diabetes risk prediction using machine learning. Our work explores how different data handling techniques and ML models can lead to more accurate and reliable early diagnosis.

    **Key Objectives:**
    - **Compare** various imputation, scaling, and modeling strategies.
    - **Validate** findings across diverse datasets (Pima Indians and a Frankfurt hospital cohort).
    - **Provide** a user-friendly tool for both prediction and research experimentation.

    The full methodology, experiments, and results are detailed in our paper. We encourage you to read it for a deeper understanding of the model's behavior.
    """)

    st.subheader("Resources")
    st.markdown("- **[View on GitHub](https://github.com/Jasmin-Patel-2110/diabetes-ml-pipeline)**")
    st.markdown("- **[Read the Full Manuscript](https://github.com/Jasmin-Patel-2110/diabetes-ml-pipeline/blob/main/Diabetes_Diagnosis_using_Machine_Learning.pdf)**")
    
    st.subheader("Authors")
    st.text("""
    - Jasmin Patel
    - Pratham Patel
    - May Patel
    - Prof. Dr. Shankar Paramar
    
    Department of Computer Engineering, Government Engineering College, Bharuch, India
    """)

    st.subheader("License")
    st.text("This project is licensed under the MIT License.")
        