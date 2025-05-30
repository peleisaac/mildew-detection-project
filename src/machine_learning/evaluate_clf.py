import streamlit as st
from src.data_management import load_pkl_file


@st.cache_data
def load_evaluation():
    return load_pkl_file("outputs/evaluation.pkl")


def display_performance_summary():
    evaluation = load_evaluation()
    perf = evaluation.get("performance_summary", {})

    st.subheader("Model Performance Summary")
    for key, value in perf.items():
        if isinstance(value, (int, float)):
            st.metric(label=key.replace("_", " ").title(), value=f"{value:.2f}")


def display_classification_report():
    evaluation = load_evaluation()
    report = evaluation.get("classification_report", {})

    st.subheader("Classification Report")
    st.json(report, expanded=False)


def display_confusion_matrix():
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    evaluation = load_evaluation()
    cm = evaluation.get("confusion_matrix")

    if cm:
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


def display_roc_curve():
    import matplotlib.pyplot as plt

    evaluation = load_evaluation()
    roc = evaluation.get("roc_data")
    if not roc:
        return

    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])
    auc = roc.get("auc", 0)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="navy")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
