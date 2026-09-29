from agents.research_agents import ResearchAgents
from crewai import Crew, Process
from tasks.research_tasks import ResearchTasks


class MultiAgentResearchCrew:

    def __init__(
        self,
        api_key: str,
        model_name: str = "groq/openai/gpt-oss-20b",
    ):
        self.api_key = api_key
        self.model_name = model_name

    def run(self, topic: str) -> str:
        agents = ResearchAgents(
            api_key=self.api_key, model_name=self.model_name
        )
        tasks = ResearchTasks()

        gen_researcher = agents.general_researcher()
        tech_researcher = agents.technical_researcher()
        checker = agents.fact_checker()
        writer = agents.report_writer()

        t1 = tasks.general_research_task(gen_researcher, topic)
        t2 = tasks.technical_research_task(tech_researcher, topic)
        t3 = tasks.fact_checking_task(checker, topic, context=[t1, t2])
        t4 = tasks.report_writing_task(writer, topic, context=[t3])

        # max_rpm=4 prevents hitting Groq's 8,000 TPM limit
        crew = Crew(
            agents=[gen_researcher, tech_researcher, checker, writer],
            tasks=[t1, t2, t3, t4],
            process=Process.sequential,
            verbose=True,
            max_rpm=4,
        )

        result = crew.kickoff()
        return str(result)
