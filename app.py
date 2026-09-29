import streamlit as st
from utils.helpers import validate_groq_api_key
from crew.research_crew import MultiAgentResearchCrew

# Streamlit Page Config
st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Multi-Agent AI Research Assistant")
    st.caption("Powered by CrewAI, Groq (Llama-3.3-70B), and DuckDuckGo Search")

    # Sidebar settings
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Select Groq Model",
        options=[
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/mixtral-8x7b-32768"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application uses a team of 4 autonomous AI agents to research, audit, and compile detailed technical reports."
    )

    # Input section
    topic = st.text_input(
        "Enter Research Topic:",
        placeholder="e.g., Quantum Computing advancements in Drug Discovery"
    )

    if st.button("Start Research", type="primary"):
        if not topic.strip():
            st.warning("Please enter a valid research topic to begin.")
            return

        # Securely fetch secret
        api_key = validate_groq_api_key()

        # Run multi-agent team
        with st.spinner("🤖 AI Agent Team is actively researching, analyzing, and writing your report... This may take 1-2 minutes."):
            try:
                research_crew = MultiAgentResearchCrew(api_key=api_key, model_name=model_name)
                final_report = research_crew.run(topic=topic)
                
                st.success("Research completed successfully!")
                st.markdown("---")
                
                # Render Report
                st.markdown(final_report)
                
                # Download Button
                st.download_button(
                    label="📥 Download Report (.md)",
                    data=final_report,
                    file_name=f"research_report_{topic.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred during agent execution: {str(e)}")

if __name__ == "__main__":
    main()
