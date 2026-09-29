from crewai import Crew, Process, LLM
from agents.research_agents import ResearchAgents
from tasks.research_tasks import ResearchTasks

# --- GROQ FIX: Monkey-patch CrewAI message formatter to strip cache_breakpoint ---
try:
    from crewai.llm import LLM as CrewLLM
    original_format_messages = getattr(CrewLLM, "_format_messages", None)
    
    if original_format_messages:
        def patched_format_messages(self, messages, *args, **kwargs):
            formatted = original_format_messages(self, messages, *args, **kwargs)
            if isinstance(formatted, list):
                clean_messages = []
                for msg in formatted:
                    if isinstance(msg, dict):
                        # Strip cache_breakpoint from message dictionary
                        clean_msg = {k: v for k, v in msg.items() if k != "cache_breakpoint"}
                        clean_messages.append(clean_msg)
                    else:
                        clean_messages.append(msg)
                return clean_messages
            return formatted
            
        CrewLLM._format_messages = patched_format_messages
except Exception:
    pass
# ----------------------------------------------------------------------------------

class MultiAgentResearchCrew:
    def __init__(self, api_key: str, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.model_name = model_name

    def run(self, topic: str) -> str:
        # Instantiate Agents & Tasks
        agents = ResearchAgents(api_key=self.api_key, model_name=self.model_name)
        tasks = ResearchTasks()

        # Build Agents
        gen_researcher = agents.general_researcher()
        tech_researcher = agents.technical_researcher()
        checker = agents.fact_checker()
        writer = agents.report_writer()

        # Build Tasks
        t1 = tasks.general_research_task(gen_researcher, topic)
        t2 = tasks.technical_research_task(tech_researcher, topic)
        t3 = tasks.fact_checking_task(checker, topic)
        t4 = tasks.report_writing_task(writer, topic)

        # Build and kickoff Crew execution sequentially
        crew = Crew(
            agents=[gen_researcher, tech_researcher, checker, writer],
            tasks=[t1, t2, t3, t4],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()
        return str(result)
