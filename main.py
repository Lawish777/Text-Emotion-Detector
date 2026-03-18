import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from time import sleep


# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("text_emotion.pkl")


pipe_lr = load_model()

# ────────────────────────────────────────────────
# Emoji mapping (expanded & cleaned)
# ────────────────────────────────────────────────
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨", "happy": "😊",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}


# ────────────────────────────────────────────────
# Prediction functions
# ────────────────────────────────────────────────
def predict_emotion(text):
    return pipe_lr.predict([text])[0]


def get_prediction_proba(text):
    return pipe_lr.predict_proba([text])[0]


# ────────────────────────────────────────────────
# Main App
# ────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Emotion Detector",
        page_icon="🧠",
        layout="wide"
    )

    # ── Header ───────────────────────────────────────
    st.title("🧠 Text Emotion Detector")
    st.markdown("Analyze the emotional tone of any text using machine learning.")

    # ── Sidebar ──────────────────────────────────────
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a pre-trained classifier to detect emotions in text.  
        Supported emotions: anger, disgust, fear, happy, joy, neutral, sad, shame, surprise.

        **Model:** Logistic Regression  
        **Features:** TF-IDF + basic preprocessing
        """)

        st.markdown("---")
        st.info("Tip: Try different sentence lengths and tones to see how the model reacts!")

    # ── Main Input Area ──────────────────────────────
    st.subheader("Enter your text below")

    with st.form(key='emotion_form', clear_on_submit=False):
        raw_text = st.text_area(
            label="",
            placeholder="Write something here... (e.g. I can't believe this happened!)",
            height=140,
            max_chars=1000,
            help="Maximum 1000 characters"
        )

        char_count = len(raw_text)
        st.caption(f"{char_count} / 1000 characters")

        submit_button = st.form_submit_button("🔍 Analyze Emotion", use_container_width=True, type="primary")

    # ── Prediction Logic ─────────────────────────────
    if submit_button:
        if not raw_text.strip():
            st.warning("Please enter some text first.")
            st.stop()

        with st.spinner("Analyzing emotion..."):
            sleep(0.6)  # small fake delay for better UX
            prediction = predict_emotion(raw_text)
            probabilities = get_prediction_proba(raw_text)

            # Get top 3 predictions
            proba_df = pd.DataFrame({
                "emotion": pipe_lr.classes_,
                "probability": probabilities
            }).sort_values("probability", ascending=False).head(3)

            proba_df["probability_pct"] = (proba_df["probability"] * 100).round(1)

        # ── Results Layout ───────────────────────────────
        col1, col2 = st.columns([3, 4])

        with col1:
            st.subheader("Input Text")
            st.markdown(f"> {raw_text}")

            st.markdown("### Predicted Emotion")
            emoji = emotions_emoji_dict.get(prediction, "❓")
            st.markdown(f"<h2 style='color:#4CAF50;'>{prediction.capitalize()} {emoji}</h2>", unsafe_allow_html=True)

            top_prob = proba_df.iloc[0]["probability_pct"]
            st.caption(f"Confidence: **{top_prob}%**")

        with col2:
            st.subheader("Probability Breakdown (Top 3)")

            # Horizontal bar chart with percentage labels
            base = alt.Chart(proba_df).encode(
                y=alt.Y("emotion:N", sort="-x", title=None),
                x=alt.X("probability_pct:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("emotion:N", legend=None),
                tooltip=["emotion", alt.Tooltip("probability_pct", format=".1f")]
            )

            bars = base.mark_bar(cornerRadius=4)

            text = bars.mark_text(
                align='left',
                baseline='middle',
                dx=5,  # space from bar end
                color='black'
            ).encode(
                text=alt.Text("probability_pct:Q", format=".1f%")  # ← fixed line
            )

            chart = (bars + text).properties(
                height=180,
                width=500
            )

            st.altair_chart(chart, use_container_width=True)
    # ── Footer ───────────────────────────────────────
    st.markdown("---")
    st.caption("Built with Streamlit • Model by Lawish • March 2026")


if __name__ == '__main__':
    main()