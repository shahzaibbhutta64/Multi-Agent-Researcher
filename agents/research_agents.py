import os
from typing import Any, Dict, List, Optional
from crewai import Agent, LLM
from tools.web_search import DuckDuckGoSearchTool


class GroqLLM(LLM):
    """Custom LLM wrapper for Groq that strips the 'cache_breakpoint' key 
    injected by CrewAI before passing messages to LiteLLM/Groq.
    """

    def _clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not messages or not isinstance(messages, list):
            return messages

        clean_list = []
        for msg in messages:
            if isinstance(msg, dict):
                # Filter out cache_breakpoint to avoid Groq validation errors
                clean_msg = {
                    k: v for k, v in msg.items() if k != "cache_breakpoint"
                }
                clean_list.append(clean_msg)
            else:
                clean_list.append(msg)
        return clean_list

    def call(
        self,
        messages: List[Dict[str, Any]],
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> str:
        cleaned = self._clean_messages(messages)
        return super().call(cleaned, callbacks=callbacks, **kwargs)

    async def acall(
        self,
        messages: List[Dict[str, Any]],
        callbacks: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> str:
        cleaned = self._clean_messages(messages)
        return await super().acall(cleaned, callbacks=callbacks, **kwargs)


class ResearchAgents:

    def __init__(
        self,
        api_key: str,
        model_name: str = "groq/openai/gpt-oss-20b",
    ):
        os.environ["GROQ_API_KEY"] = api_key
        self.llm = GroqLLM(model=model_name, api_key=api_key)
        self.search_tool = DuckDuckGoSearchTool()

    def general_researcher(self) -> Agent:
        return Agent(
            role="General Researcher",
            goal=(
                "Discover concise background facts, key statistics, and high-level"
                " summaries on the user's topic."
            ),
            backstory=(
                "You are an investigative research journalist focused on brief,"
                " accurate high-level facts."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,            # Limits tool search loops
            max_execution_time=90, # Prevents long hanging execution
        )

    def technical_researcher(self) -> Agent:
        return Agent(
            role="Technical and Critical Researcher",
            goal=(
                "Investigate technical mechanisms, architectures, and implementation risks."
            ),
            backstory=(
                "You are a concise systems analyst focusing purely on structural"
                " mechanics and feasibility."
            ),
            tools=[self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=90,
        )

    def fact_checker(self) -> Agent:
        return Agent(
            role="Research Analyst and Fact Checker",
            goal=(
                "Verify claims, remove duplicate information, and resolve contradictions."
            ),
            backstory=(
                "You are a lead fact-checker who condenses and validates information concise and clear."
            ),
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2,
        )

    def report_writer(self) -> Agent:
        return Agent(
            role="Senior Technical Report Writer",
            goal=(
                "Synthesize verified findings into a clean Markdown research report."
            ),
            backstory=(
                "You are an expert technical editor who writes clear executive summaries and reports."
            ),
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            max_iter=2,
        )
