from crewai import Task, Agent

class ResearchTasks:
    def general_research_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Conduct a broad initial web search on the topic: '{topic}'.\n"
                "1. Search for general background, fundamental concepts, key facts, and recent developments.\n"
                "2. Identify notable statistics, key stakeholders, or organizations involved.\n"
                "3. Collect credible source URLs along with your findings."
            ),
            expected_output=(
                "A bulleted summary of general findings, key facts, background context, "
                "and an explicit list of gathered source URLs."
            ),
            agent=agent
        )

    def technical_research_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Perform a technical and critical investigation into: '{topic}'.\n"
                "1. Search for technical mechanics, engineering or structural details, benefits, and challenges.\n"
                "2. Search for real-world applications, concrete examples, and implementation risks.\n"
                "3. Record all source URLs discovered."
            ),
            expected_output=(
                "A detailed breakdown covering technical implementation details, key benefits, "
                "limitations, risks, and primary source URLs."
            ),
            agent=agent
        )

    def fact_checking_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Synthesize and audit the research outputs for the topic: '{topic}'.\n"
                "1. Cross-reference findings from the General Researcher and Technical Researcher.\n"
                "2. Remove duplicate information and flag unverified or contradictory statements.\n"
                "3. Select the most accurate, high-impact facts and consolidate verified source URLs."
            ),
            expected_output=(
                "A consolidated, verified synthesis of facts, structured logically with verified source URLs "
                "and notes on any uncertainties."
            ),
            agent=agent
        )

    def report_writing_task(self, agent: Agent, topic: str) -> Task:
        return Task(
            description=(
                f"Write a final comprehensive research report for the topic: '{topic}' using the verified research synthesis.\n"
                "Structure the final report with the following Markdown headings:\n"
                "- # Executive Summary\n"
                "- ## Introduction & Background\n"
                "- ## Key Findings & Technical Details\n"
                "- ## Benefits & Opportunities\n"
                "- ## Risks & Challenges\n"
                "- ## Real-World Applications\n"
                "- ## Conclusion\n"
                "- ## Sources & References (Include hyperlinked or listed source URLs)\n\n"
                "Ensure tone is professional, clear, and objective."
            ),
            expected_output=(
                "A full, publication-ready Markdown research report with exact headings and complete source URLs."
            ),
            agent=agent
        )
