# =============================================================================
# src/ai_news_crew/agents.py — Agent definitions for the AI News Crew
# =============================================================================
#
# WHAT IS A CREWAI AGENT?
#   An Agent is an autonomous AI worker with a:
#     - ROLE      : job title that shapes reasoning style
#     - GOAL      : specific outcome it must achieve
#     - BACKSTORY : narrative context influencing its persona
#     - TOOLS     : capabilities it can invoke during reasoning
#     - LLM       : the underlying model powering its thinking
#
# THE RACT REASONING LOOP (Reason + Act):
#   Agents don't just respond once — they loop:
#     1. THOUGHT      → "I need to find AI news..."
#     2. ACTION       → call SerperDevTool("latest AI news 2026")
#     3. OBSERVATION  → [search results]
#     4. THOUGHT      → "I should read that article..."
#     5. ACTION       → call ScrapeWebsiteTool("https://...")
#     6. OBSERVATION  → [article text]
#     7. FINAL ANSWER → [structured research output]
#
# THE TWO AGENTS HERE:
#   1. news_researcher — searches + scrapes to gather raw findings
#   2. news_writer     — synthesises findings into a Markdown report
#
# KEY PARAMETERS:
#   verbose=True  → prints every thought/action (invaluable for learning!)
#   max_iter      → limits ReAct loops to prevent infinite searching
#   tools=[]      → writer has no tools: clear separation of concerns
#
# =============================================================================

from crewai import Agent, LLM

from .tools import get_scrape_tool, get_search_tool


def create_news_researcher(llm: LLM) -> Agent:
    """
    The 'legwork' agent: searches the web and scrapes articles to produce
    a structured set of AI/LLM news findings.

    Args:
        llm: Configured LLM instance from config.get_llm().
    Returns:
        Agent: The configured News Researcher agent.
    """
    return Agent(
        role="Senior AI Research Analyst",

        goal=(
            "Discover and investigate the most significant AI and LLM developments "
            "from the past 7 days. Find 8-10 distinct stories covering model releases, "
            "research breakthroughs, product launches, and industry news. "
            "For each story capture: headline, source URL, publication date, summary."
        ),

        backstory=(
            "You are a seasoned technology journalist with 15 years covering "
            "artificial intelligence. You have an encyclopaedic knowledge of the AI "
            "landscape and a journalist's instinct for what matters. You are meticulous "
            "about citing sources, sceptical of hype, and always verify information "
            "before reporting it. A story without a source is just gossip."
        ),

        # Tools list defines what this agent CAN do.
        # Agents cannot use tools not listed here.
        tools=[get_search_tool(), get_scrape_tool()],

        llm=llm,

        # verbose=True → see every thought and tool call in real time.
        # This is the best way to understand how agentic AI actually works.
        verbose=True,

        # Researcher does not hand off work — it owns its research task fully.
        allow_delegation=False,

        # Allow up to 15 search/scrape cycles before forcing a final answer.
        max_iter=15,
    )


def create_news_writer(llm: LLM) -> Agent:
    """
    The 'craft' agent: takes researcher findings and writes a polished
    Markdown summary report. Has no tools — works from context only.

    Args:
        llm: Configured LLM instance from config.get_llm().
    Returns:
        Agent: The configured News Writer agent.
    """
    return Agent(
        role="AI Technology Reporter",

        goal=(
            "Transform raw research findings into a compelling, well-structured "
            "Markdown report. Include clear sections with emoji headings, a brief "
            "executive summary, story breakdowns, trend analysis, and a sources list. "
            "The report should be useful to both technical and non-technical readers."
        ),

        backstory=(
            "You are an award-winning technology reporter known for making complex "
            "AI topics accessible to a broad audience. Your weekly AI digest is "
            "trusted by thousands of developers, researchers, and business leaders. "
            "You are an expert at Markdown formatting and structure content so both "
            "the time-pressed executive and the curious developer find value. "
            "You never invent facts — you only report what the research confirms."
        ),

        # No tools: the writer synthesises from context, does not browse.
        # This mirrors real-world specialisation: researchers gather, writers craft.
        tools=[],

        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )
