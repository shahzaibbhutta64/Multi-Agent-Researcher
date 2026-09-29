import os
import litellm
from crewai import Agent, LLM
from tools.web_search import DuckDuckGoSearchTool

# Tell LiteLLM to automatically strip unsupported parameters (like cache_breakpoint) for Groq
litellm.drop_params = True

class ResearchAgents:
    def __init__(self, api_key: str, model_name: str = "groq/llama-3.3-70b-versatile"):
        # Set environment variable so LiteLLM and CrewAI can locate the key automatically
        os.environ["GROQ_API_KEY"] = api_key

        # Initialize CrewAI LLM wrapper
        self.llm = LLM(
            model=model_name,
            api_key=api_key
        )
        self.search_tool = DuckDuckGoSearchTool()

    def general_researcher(self) -> Agent:
        return Agent(
            role="General Researcher",
            goal="Discover comprehensive background facts, statistics, and high-level summaries on the user's topic.",
            backstory=(
                "You are an experienced investigative research journalist. Your specialty is gathering broad, "
                "reliable information and discovering relevant background data across credible web sources."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def technical_researcher(self) -> Agent:
        return Agent(
            role="Technical and Critical Researcher",
            goal="Investigate technical mechanisms, architecture, real-world applications, risks, and limitations.",
            backstory=(
                "You are an analytical domain expert and systems analyst. You dive into technical mechanics, "
                "evaluate claim feasibility, uncover potential failure modes, and identify concrete implementation details."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def fact_checker(self) -> Agent:
        return Agent(
            role="Research Analyst and Fact Checker",
            goal="Verify claims from research findings, eliminate duplicated information, and resolve conflicting information.",
            backstory=(
                "You are a meticulous lead researcher and fact-checker. You filter noise, identify contradictory statements, "
                "validate information against credible source URLs, and curate only high-confidence insights."
            ),
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def report_writer(self) -> Agent:
        return Agent(
            role="Senior Technical Report Writer",
            goal="Synthesize verified research findings into a clear, beautifully structured report with explicit source citations.",
            backstory=(
                "You are an expert technical editor. You write executive summaries and full structured reports that translate "
                "complex insights into clear sections while maintaining clear source URL attributions."
            ),
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
