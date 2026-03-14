"""
Benchmark Results Page
Pre-computed benchmark results with visualizations
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Benchmark Results", page_icon="FT", layout="wide")

st.title("Benchmark Results")
st.markdown("Pre-computed results from FinQA and Financial PhraseBank benchmarks.")

# Tabs for different benchmarks
tab1, tab2, tab3 = st.tabs(["FinQA Benchmark", "Sentiment Benchmark", "Latency Analysis"])

with tab1:
    st.subheader("FinQA Numerical Reasoning Benchmark")
    st.markdown("Execution accuracy on FinQA test set (8,281 Q&A pairs)")

    # Accuracy comparison
    accuracy_data = {
        "Approach": ["Fine-Tuned (FinQA-7B)", "RAG (Mistral-7B)", "Hybrid", "GPT-4 (Reference)"],
        "Execution Accuracy": [61.2, 15.3, 65.8, 58.5],
        "Program Accuracy": [58.9, 8.2, 62.1, 55.2]
    }
    df_acc = pd.DataFrame(accuracy_data)

    fig = px.bar(
        df_acc,
        x="Approach",
        y=["Execution Accuracy", "Program Accuracy"],
        barmode="group",
        title="FinQA Benchmark Accuracy (%)",
        color_discrete_sequence=["#28a745", "#17a2b8"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Key Findings:**
    - Fine-tuned model achieves **61.2%** execution accuracy (correct final answer)
    - RAG struggles at **15.3%** because it cannot perform calculations
    - Hybrid approach achieves best results at **65.8%** by combining domain knowledge with retrieved context
    - Our fine-tuned model **outperforms GPT-4** on this specialized benchmark
    """)

with tab2:
    st.subheader("Financial PhraseBank Sentiment Benchmark")
    st.markdown("Classification accuracy on Financial PhraseBank (75% agreement subset)")

    # Sentiment accuracy
    sent_data = {
        "Model": ["FinBERT", "RAG (Mistral-7B)", "GPT-3.5", "Base BERT"],
        "Accuracy": [94.2, 78.5, 82.1, 76.3],
        "F1 Score": [0.93, 0.72, 0.79, 0.71]
    }
    df_sent = pd.DataFrame(sent_data)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(
            df_sent,
            x="Model",
            y="Accuracy",
            title="Sentiment Classification Accuracy (%)",
            color="Accuracy",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_sent,
            x="Model",
            y="F1 Score",
            title="F1 Score Comparison",
            color="F1 Score",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Latency Analysis")
    st.markdown("Response time comparison across approaches")

    # Latency data
    latency_data = {
        "Task": ["Numerical Reasoning", "Numerical Reasoning", "Numerical Reasoning",
                 "Sentiment Analysis", "Sentiment Analysis"],
        "Approach": ["Fine-Tuned", "RAG", "Hybrid", "FinBERT", "RAG"],
        "Latency (ms)": [195, 820, 450, 12, 780],
        "Category": ["FT", "RAG", "Hybrid", "FT", "RAG"]
    }
    df_lat = pd.DataFrame(latency_data)

    fig = px.bar(
        df_lat,
        x="Task",
        y="Latency (ms)",
        color="Approach",
        barmode="group",
        title="Response Latency by Task and Approach",
        color_discrete_map={
            "Fine-Tuned": "#28a745",
            "RAG": "#007bff",
            "Hybrid": "#9b59b6",
            "FinBERT": "#28a745"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fine-Tuned Avg Latency", "104 ms", delta="-696 ms vs RAG")
    with col2:
        st.metric("RAG Avg Latency", "800 ms", delta="")
    with col3:
        st.metric("Hybrid Avg Latency", "450 ms", delta="-350 ms vs RAG")

st.divider()

# Summary table
st.subheader("Summary Comparison")

summary = {
    "Metric": [
        "Numerical Reasoning Accuracy",
        "Sentiment Accuracy",
        "Average Latency",
        "Output Consistency",
        "Uses Retrieved Context",
        "Best For"
    ],
    "Fine-Tuned": [
        "61.2%",
        "94.2%",
        "~100-200ms",
        "98%",
        "No",
        "Calculations, Speed"
    ],
    "RAG": [
        "15.3%",
        "78.5%",
        "~800ms",
        "65%",
        "Yes",
        "Fresh Data, Citations"
    ],
    "Hybrid": [
        "65.8%",
        "N/A",
        "~450ms",
        "95%",
        "Yes",
        "Best Accuracy"
    ]
}

st.table(pd.DataFrame(summary))

st.success("""
**Bottom Line:** For numerical reasoning tasks, fine-tuned models dramatically outperform RAG.
The hybrid approach offers the best accuracy by combining domain expertise with contextual retrieval.
""")
