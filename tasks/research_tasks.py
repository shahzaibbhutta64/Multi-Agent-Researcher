from crewai import Task, Agent
from typing import List, Optional

class ResearchTasks:
    def general_research_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Conduct a brief initial web search on the topic: '{topic}'.\n"
                "1. Gather general background and 3-5 core key facts.\n"
                "2. Keep research concise to avoid context blowup.\n"
                "3. Collect credible source URLs."
            ),
            expected_output=(
                "A bulleted summary of key facts, background context, and gathered source URLs."
            ),
            agent=agent
        )

    def technical_research_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Perform a targeted technical investigation into: '{topic}'.\n"
                "1. Search for core mechanics, major benefits, and key challenges.\n"
                "2. Summarize findings concisely and record source URLs."
            ),
            expected_output=(
                "A concise breakdown covering technical implementation details, key benefits, risks, and source URLs."
            ),
            agent=agent
        )

    def fact_checking_task(self, agent: Agent, topic: str, context: Optional[List[Task]] = None) -> Task:
        return Task(
            description=(
                f"Audit and synthesize research outputs for: '{topic}'.\n"
                "1. Cross-reference research findings from prior tasks.\n"
                "2. Eliminate duplicate points and flag unverified claims.\n"
                "3. Consolidate a clean, verified list of facts and source URLs."
            ),
            expected_output=(
                "A consolidated, verified synthesis of facts structured logically with source URLs."
            ),
            agent=agent,
            context=context
        )

    def report_writing_task(self, agent: Agent, topic: str, context: Optional[List[Task]] = None) -> Task:
        return Task(
            description=(
                f"Write a final comprehensive research report for '{topic}' using the verified research synthesis.\n"
                "Structure the final report with standard Markdown headings:\n"
                "- # Executive Summary\n"
                "- ## Introduction & Background\n"
                "- ## Key Findings & Technical Details\n"
                "- ## Benefits & Opportunities\n"
                "- ## Risks & Challenges\n"
                "- ## Real-World Applications\n"
                "- ## Conclusion\n"
                "- ## Sources & References\n"
            ),
            expected_output=(
                "A complete Markdown research report with exact headings and complete source URLs."
            ),
            agent=agent,
            context=context
        )
