from crewai.tools import BaseTool
from duckduckgo_search import DDGS
import json

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Web Search"
    description: str = (
        "Searches the web using DuckDuckGo. Input should be a search query string. "
        "Returns a JSON string containing title, href (URL), and body (snippet) for relevant results."
    )

    def _run(self, query: str) -> str:
        """Executes search and handles errors gracefully."""
        try:
            results = []
            with DDGS() as ddgs:
                # Perform web search returning top 5 results
                raw_results = list(ddgs.text(query, max_results=5))
                for r in raw_results:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            
            if not results:
                return "No search results found for the query."
            
            return json.dumps(results, indent=2)
        except Exception as e:
            # Graceful error string return to avoid crashing the agent execution
            return f"Search execution encountered an issue: {str(e)}"
