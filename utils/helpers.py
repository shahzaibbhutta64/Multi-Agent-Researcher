import streamlit as st

def validate_groq_api_key() -> str:
    """Safely reads the Groq API key from Streamlit Secrets."""
    if "GROQ_API_KEY" not in st.secrets or not st.secrets["GROQ_API_KEY"].strip():
        st.error(
            "⚠️ Groq API key missing! Please configure `GROQ_API_KEY` in Streamlit Cloud Secrets or `.streamlit/secrets.toml`."
        )
        st.stop()
    return st.secrets["GROQ_API_KEY"].strip()
