# =============================================================================
# src/ai_news_crew/tools.py — Tool configuration for CrewAI agents
# =============================================================================
#
# PURPOSE OF THIS FILE:
#   Define and configure the tools that agents can use to interact with
#   the outside world. Without tools, agents can only reason with their
#   training knowledge — tools give them real-time capabilities.
#
# =============================================================================
#
# WHAT ARE TOOLS IN CREWAI?
#
#   Tools are Python functions (or classes) that agents can choose to call
#   during their reasoning process. An agent decides WHEN and WHETHER to use
#   a tool based on the task at hand — this is called the ReAct pattern:
#
#     ReAct Loop (Reason + Act):
#       1. THOUGHT:   "I need to find today's AI news..."
#       2. ACTION:    Call SerperDevTool with query "latest AI LLM news 2026"
#       3. OBSERVATION: [search results returned]
#       4. THOUGHT:   "I should read that first article in detail..."
#       5. ACTION:    Call ScrapeWebsiteTool with URL from search result
#       6. OBSERVATION: [article content returned]
#       7. THOUGHT:   "I have enough information now. Let me write my report."
#       8. FINAL ANSWER: [structured research output]
#
#   This loop repeats until the agent decides it has enough information
#   or hits the max_iterations limit.
#
# =============================================================================
#
# TOOLS USED IN THIS PROJECT:
#
#   1. SerperDevTool  — Searches Google via the Serper.dev API
#      - Provides real-time web search results (title, URL, snippet)
#      - Also includes news-specific results (published date, source)
#      - Free tier: 2,500 queries/month at https://serper.dev
#      - Reads from: SERPER_API_KEY environment variable
#
#   2. ScrapeWebsiteTool — Fetches and extracts text from a webpage
#      - Given a URL, downloads the page and returns its text content
#      - The agent uses this to read full articles found via search
#      - No API key needed — uses HTTP requests directly
#
# =============================================================================
#
# HOW TO CREATE A CUSTOM TOOL:
#
#   Option A — Subclass BaseTool (for complex tools with validation):
#
#       from crewai.tools import BaseTool
#       from pydantic import BaseModel, Field
#
#       class MyToolInput(BaseModel):
#           url: str = Field(description="The URL to process")
#
#       class MyCustomTool(BaseTool):
#           name: str = "My Custom Tool"
#           description: str = "Does something useful given a URL"
#           args_schema: type[BaseModel] = MyToolInput
#
#           def _run(self, url: str) -> str:
#               # Your logic here — must return a string
#               return f"Processed: {url}"
#
#   Option B — @tool decorator (for simple tools):
#
#       from crewai import tool
#
#       @tool("Fetch RSS Feed")
#       def fetch_rss(feed_url: str) -> str:
#           """Fetches and parses an RSS feed, returning recent headlines."""
#           import feedparser
#           feed = feedparser.parse(feed_url)
#           headlines = [entry.title for entry in feed.entries[:10]]
#           return "\n".join(headlines)
#
# =============================================================================

from crewai_tools import ScrapeWebsiteTool, SerperDevTool


def get_search_tool() -> SerperDevTool:
    """
    Create and return a configured SerperDevTool instance.

    SerperDevTool automatically reads the SERPER_API_KEY environment variable.
    Each call to this tool costs one API credit from your Serper.dev account.

    The `n_results` parameter controls how many search results are returned
    per query. More results = more context for the agent, but also more tokens
    used in the LLM's context window.

    Returns:
        SerperDevTool: Ready-to-use Google search tool for agents.
    """
    # n_results=10 returns 10 search results per query.
    # The agent can make multiple search queries if needed.
    return SerperDevTool(n_results=10)


def get_scrape_tool() -> ScrapeWebsiteTool:
    """
    Create and return a configured ScrapeWebsiteTool instance.

    ScrapeWebsiteTool takes a URL and returns the page's text content.
    It strips HTML tags, navigation, and other boilerplate, returning
    just the readable text. This lets agents read full article content
    rather than relying on short search snippets.

    Returns:
        ScrapeWebsiteTool: Ready-to-use web scraping tool for agents.
    """
    return ScrapeWebsiteTool()
