import streamlit as st
from utils.helpers import validate_groq_api_key
from crew.research_crew import MultiAgentResearchCrew

# Fixed background model
DEFAULT_MODEL = "groq/openai/gpt-oss-20b"

# Streamlit Page Config
st.set_page_config(
    page_title="Multi-Agent AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.caption("Autonomous Multi-Agent Intelligence System")
        st.markdown("---")

        # Application Purpose
        st.subheader("🎯 Application Purpose")
        st.write(
            "An enterprise-grade autonomous research system designed to conduct "
            "deep web inquiries, cross-verify information, and draft publication-ready "
            "technical reports without manual intervention."
        )

        st.markdown("---")

        # Input & Output Guide
        st.subheader("📥 Input Required")
        st.info("• **Research Topic**: Any technical, scientific, or industry domain concept.")

        st.subheader("📤 Output Generated")
        st.success(
            "• **Structured Markdown Report** covering Executive Summary, Technical Details, "
            "Risks, Real-World Applications, and Verified Source Citations."
        )

        st.markdown("---")

        # Agent Architecture Breakdown
        st.subheader("🤖 Autonomous Agent Team")
        st.markdown(
            """
            1. **General Researcher**: Scours live web data for general background and key stats.
            2. **Technical Analyst**: Investigates core mechanisms, architecture, and risks.
            3. **Fact Checker**: Audits findings, resolves contradictions, and validates sources.
            4. **Report Writer**: Synthesizes verified data into a formal publication-ready report.
            """
        )

        st.markdown("---")
        st.caption("Developed by Shahzaib | GitHub: shahzaibbhutta64")

    # --- MAIN PAGE UI ---
    st.title("🤖 Autonomous Multi-Agent Research Assistant")
    st.markdown(
        "Powered by **CrewAI**, **Groq (GPT-OSS 20B)**, and **DuckDuckGo Live Web Search**."
    )

    # Top metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Autonomous Agents", value="4 Specialized Roles")
    with col2:
        st.metric(label="Search Engine Integration", value="DuckDuckGo Search")
    with col3:
        st.metric(label="LLM Engine", value="GPT-OSS 20B (Groq)")

    st.markdown("---")

    # Input section
    st.subheader("🔍 Initiate Research Project")
    topic = st.text_input(
        "Specify Research Topic or Question:",
        placeholder="e.g., AI Impact on HSE, Quantum Computing advances, or Renewable Energy Storage",
        help="Enter any complex topic to run full multi-agent web investigation."
    )

    if st.button("🚀 Start Deep Research", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("⚠️ Please enter a valid research topic before proceeding.")
            return

        # Securely fetch secret
        api_key = validate_groq_api_key()

        # Run multi-agent team
        with st.spinner("🔄 Agent team is searching the web, cross-verifying facts, and compiling your report..."):
            try:
                research_crew = MultiAgentResearchCrew(api_key=api_key, model_name=DEFAULT_MODEL)
                final_report = research_crew.run(topic=topic)
                
                st.success("✅ Research completed successfully!")
                st.markdown("---")
                
                # Render Report
                st.markdown(final_report)
                
                # Download Button
                st.download_button(
                    label="📥 Download Complete Report (.md)",
                    data=final_report,
                    file_name=f"research_report_{topic.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ An error occurred during agent execution: {str(e)}")

if __name__ == "__main__":
    main()
