# 🤖 AI News Crew — Agentic AI Tutorial with CrewAI + AWS Bedrock

> A hands-on tutorial project that builds a **multi-agent AI system** to automatically research, summarise, and report on the latest AI/LLM news. Built with [CrewAI](https://crewai.com), powered by [AWS Bedrock](https://aws.amazon.com/bedrock/), and managed with [uv](https://docs.astral.sh/uv/).

---

## 📖 Table of Contents

1. [What is Agentic AI?](#1-what-is-agentic-ai)
2. [What is CrewAI?](#2-what-is-crewai)
3. [What Does This Project Do?](#3-what-does-this-project-do)
4. [Project Architecture](#4-project-architecture)
5. [Prerequisites](#5-prerequisites)
6. [Quickstart](#6-quickstart)
7. [Configuration Reference](#7-configuration-reference)
8. [Understanding the Code](#8-understanding-the-code)
9. [How It Works — The ReAct Loop](#9-how-it-works--the-react-loop)
10. [Sample Output](#10-sample-output)
11. [Troubleshooting](#11-troubleshooting)
12. [Extending the Project](#12-extending-the-project)
13. [Key Concepts Glossary](#13-key-concepts-glossary)

---

## 1. What is Agentic AI?

Traditional AI applications follow a fixed pattern: you send a prompt, you get a response. **Agentic AI** is different — an agent can:

- **Plan** a sequence of steps to reach a goal
- **Use tools** to interact with the real world (web search, databases, APIs)
- **Observe** the results of its actions and adjust
- **Loop** through reasoning until it decides it has a satisfactory answer

This is sometimes called the **ReAct pattern** (Reason + Act). Instead of one prompt → one response, you get:

```
THOUGHT  → "I need to find today's AI news"
ACTION   → [calls SerperDevTool: "latest AI news October 2026"]
OBSERVE  → [receives 10 search results]
THOUGHT  → "The Anthropic announcement looks important, I should read it"
ACTION   → [calls ScrapeWebsiteTool: "https://anthropic.com/..."]
OBSERVE  → [receives full article text]
THOUGHT  → "I have enough information for a thorough report now"
ANSWER   → [produces structured research findings]
```

This loop can repeat many times — the agent keeps working until it decides its goal is met or hits a limit. That's what makes it "agentic": it has agency over its own process.

---

## 2. What is CrewAI?

[CrewAI](https://crewai.com) is an open-source Python framework for building **multi-agent AI systems**. It provides:

| Concept | What it is |
|---|---|
| **Agent** | An autonomous AI worker with a role, goal, backstory, and tools |
| **Task** | A specific piece of work assigned to an agent, with expected output |
| **Tool** | A callable function an agent can use (search, scrape, read file, etc.) |
| **Crew** | A team of agents working together through a defined process |
| **Process** | How tasks are executed — sequential or hierarchical |
| **LLM** | The underlying language model powering an agent's reasoning |

### Why CrewAI?

- **Framework independence**: Built from scratch, not tied to LangChain
- **Multi-provider LLM support**: Works with OpenAI, Anthropic, AWS Bedrock, Google, and 100+ others via LiteLLM
- **Batteries included**: Pre-built tools for web search, scraping, file I/O, and more
- **Production ready**: Used by enterprises at scale, with tracing, memory, and flow support
- **Great for learning**: The verbose output shows you exactly how agents reason

### CrewAI vs Just Prompting an LLM

| Just Prompting | CrewAI Agents |
|---|---|
| One prompt, one answer | Multi-step reasoning loops |
| No tool access | Can search, scrape, read files |
| Fixed context | Dynamically gathers information |
| Single model call | Multiple model calls with context |
| You write the logic | Agent decides its own steps |

---

## 3. What Does This Project Do?

This project runs a **two-agent crew** that:

1. **🔍 Research Agent** (`Senior AI Research Analyst`)
   - Searches Google for the latest AI/LLM news using Serper.dev
   - Reads full articles for important stories using web scraping
   - Produces a structured research report with sources, dates, and summaries

2. **✍️ Writer Agent** (`AI Technology Reporter`)
   - Receives the research as context
   - Writes a polished, structured Markdown report
   - Automatically saves it to `output/ai_news_summary.md`

The final output is a Markdown file structured like this:

```markdown
# 🤖 AI & LLM News Digest — 2026-10-28

## 📋 Executive Summary
This week saw major announcements from...

## 🔥 Top Stories
### Anthropic Releases Claude 4 with 1M Token Context
...

## 🔗 Sources
1. [Anthropic Blog](https://anthropic.com/...)
```

---

## 4. Project Architecture

```
agentic-ui-tut/
│
├── 📄 README.md                     ← You are here
├── ⚙️  pyproject.toml               ← uv project config & dependencies
├── 🔒 .env.example                  ← Environment variable template
├── 🚫 .gitignore                    ← Keeps secrets out of git
│
├── src/
│   └── ai_news_crew/                ← Main Python package
│       ├── __init__.py              ← Package marker
│       ├── config.py                ← LLM setup (AWS Bedrock) + env loading
│       ├── tools.py                 ← Tool definitions (search + scrape)
│       ├── agents.py                ← Agent definitions (Researcher + Writer)
│       ├── tasks.py                 ← Task definitions (Research + Write)
│       ├── crew.py                  ← Crew assembly + execution
│       └── main.py                  ← CLI entry point
│
└── output/
    └── ai_news_summary.md           ← Generated report (created at runtime)
```

### Data Flow

```
  uv run ai-news-crew
         │
         ▼
  config.py: Load .env → Validate → Build LLM
         │
         ▼
  ┌──────────────── CREW ─────────────────────┐
  │                                           │
  │  TASK 1 — Senior AI Research Analyst      │
  │    Tools: SerperDevTool + ScrapeWebsite   │
  │    [THINK → SEARCH → OBSERVE → SCRAPE]   │
  │    Output: Structured research findings   │
  │                  │                        │
  │             context passed automatically  │
  │                  ▼                        │
  │  TASK 2 — AI Technology Reporter          │
  │    Tools: (none — works from context)     │
  │    [WRITE → FORMAT → STRUCTURE]           │
  │    Output: output/ai_news_summary.md ──► 📄
  └───────────────────────────────────────────┘
```

### How AWS Bedrock Fits In

```
  CrewAI Agent
      │  crewai.LLM(model="bedrock/anthropic.claude-...")
      ▼
  LiteLLM (bundled with CrewAI)
      │  Translates call to Bedrock API format
      ▼
  boto3 (AWS SDK for Python)
      │  Signs request with AWS credentials
      │  (from env vars / ~/.aws/credentials / IAM role)
      ▼
  AWS Bedrock API → Model Response → back to Agent
```

---

## 5. Prerequisites

### 5.1 Python & uv

You need **Python 3.11+** and **uv** installed:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version       # should show 0.x.x
python3 --version  # should show 3.11+
```

### 5.2 AWS Account with Bedrock Access

1. Sign in to the [AWS Console](https://console.aws.amazon.com)
2. Navigate to **Amazon Bedrock** → **Model access** → **Manage model access**
3. Select and request access to your desired model (recommended: **Claude 3.5 Sonnet v2** in `us-east-1`)
   - ⚠️ Access is **per-region** — make sure you're in the region you'll set in `.env`
   - Access is usually granted instantly for most models
4. Ensure your IAM user has these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ],
    "Resource": "*"
  }]
}
```

5. Generate an **Access Key ID** and **Secret Access Key** for your IAM user

> 💡 **Tip**: If `~/.aws/credentials` is already configured (via `aws configure`), you don't need to set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env` — boto3 will find them automatically.

### 5.3 Serper.dev API Key

Serper.dev provides the Google Search API used by the research agent.

1. Go to [serper.dev](https://serper.dev)
2. Click **Sign Up** — **free tier: 2,500 queries/month, no credit card required**
3. Copy your API key from [serper.dev/api-key](https://serper.dev/api-key)

Each run of this project uses approximately **5–15 search queries**.

---

## 6. Quickstart

```bash
# 1. Enter the project directory
cd /path/to/agentic-ui-tut

# 2. Copy the environment template
cp .env.example .env

# 3. Edit .env with your real credentials
nano .env
# Fill in: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
#          AWS_DEFAULT_REGION, BEDROCK_MODEL_ID, SERPER_API_KEY

# 4. Install all dependencies
uv sync

# 5. Run the crew!
uv run ai-news-crew
```

The crew will run for **5–15 minutes** and save the report to `output/ai_news_summary.md`.

### What You'll See

With `verbose=True` (the default), you'll see the agents' full reasoning in real time:

```
🔧 Building AI News Crew...
🤖 Configuring LLM: bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
🚀 Starting crew execution...

[Senior AI Research Analyst] Starting task...

  Thought: I need to search for the latest AI news.
  Action: Search the internet
  Action Input: {"query": "latest AI LLM news October 2026"}
  Observation: [10 search results with titles, snippets, URLs]

  Thought: The Anthropic announcement looks significant. Let me read it fully.
  Action: Read website content
  Action Input: {"website_url": "https://anthropic.com/blog/..."}
  Observation: [full article text]

  ... [more reasoning loops] ...

  Final Answer: [structured research report with 10 stories]

[AI Technology Reporter] Starting task...
  ... [writing the Markdown report] ...

✅ Report saved to: output/ai_news_summary.md
```

This verbose output is one of the best learning tools — you can watch exactly how the agent thinks, what it searches for, and why it makes each decision.

---

## 7. Configuration Reference

All configuration lives in `.env`. Copy `.env.example` to `.env` to get started.

### AWS Settings

| Variable | Required | Description | Example |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | ✅* | IAM user access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | ✅* | IAM user secret key | `wJalrXUtnFEMI/...` |
| `AWS_DEFAULT_REGION` | ✅ | AWS region with Bedrock enabled | `us-east-1` |
| `AWS_PROFILE` | Optional | Named profile from `~/.aws/credentials` | `my-dev-profile` |

*Not required if using `~/.aws/credentials` or an IAM role.

### Bedrock Model Settings

| Variable | Required | Default |
|---|---|---|
| `BEDROCK_MODEL_ID` | ✅ | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `BEDROCK_TEMPERATURE` | Optional | `0.7` |

### Available Bedrock Models

| Provider | Model | `BEDROCK_MODEL_ID` value | Notes |
|---|---|---|---|
| Anthropic | Claude 3.5 Sonnet v2 | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | ⭐ Recommended |
| Anthropic | Claude 3.5 Haiku | `bedrock/anthropic.claude-3-5-haiku-20241022-v1:0` | Faster, cheaper |
| Anthropic | Claude 3 Opus | `bedrock/anthropic.claude-3-opus-20240229-v1:0` | Most powerful |
| Amazon | Nova Pro | `bedrock/amazon.nova-pro-v1:0` | Good value |
| Amazon | Nova Lite | `bedrock/amazon.nova-lite-v1:0` | Very cheap |
| Meta | Llama 3 70B | `bedrock/meta.llama3-70b-instruct-v1:0` | Open source |
| Mistral | Mistral Large | `bedrock/mistral.mistral-large-2402-v1:0` | European provider |

> ⚠️ You must enable model access in the AWS Bedrock console first. Not all models are available in all regions.

### Search & Output Settings

| Variable | Required | Default |
|---|---|---|
| `SERPER_API_KEY` | ✅ | (none) |
| `OUTPUT_FILE` | Optional | `output/ai_news_summary.md` |

---

## 8. Understanding the Code

### `config.py` — The Foundation

**Key concept: LiteLLM as a universal LLM adapter**

```python
from crewai import LLM

llm = LLM(
    model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature=0.7,
    max_tokens=4096,
)
```

`crewai.LLM` wraps LiteLLM, which uses boto3 to call AWS Bedrock. The `"bedrock/"` prefix tells LiteLLM which provider adapter to use. Your AWS credentials come from environment variables automatically via boto3's credential chain.

### `tools.py` — Agent Capabilities

**Key concept: Tools give agents real-world abilities**

```python
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

search_tool = SerperDevTool(n_results=10)  # searches Google via Serper.dev
scrape_tool = ScrapeWebsiteTool()           # reads a webpage's full text
```

Tools are functions agents can call during the ReAct loop. `SerperDevTool` uses `SERPER_API_KEY` automatically. The agent decides *when* and *whether* to use each tool.

### `agents.py` — The Workers

**Key concept: Role/Goal/Backstory shape LLM behaviour**

```python
researcher = Agent(
    role="Senior AI Research Analyst",
    goal="Find the top AI/LLM news stories with sources...",
    backstory="You are a seasoned journalist with 15 years covering AI...",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,  # ← prints every thought and action — great for learning!
)
```

The `role`, `goal`, and `backstory` are injected into the system prompt and dramatically shape how the model reasons. Try changing the backstory and see how the outputs differ!

### `tasks.py` — The Work Orders

**Key concept: Tasks chain via `context=[]` and auto-save via `output_file`**

```python
research_task = Task(
    description="Search for AI news from the past 7 days...",
    expected_output="A structured report with 8-10 stories...",
    agent=researcher,
)

write_task = Task(
    description="Write a Markdown report from the research...",
    context=[research_task],         # ← researcher's output injected here auto
    output_file="output/report.md",  # ← CrewAI saves the result to disk
    agent=writer,
)
```

`context=[research_task]` makes CrewAI automatically append the researcher's final answer into the writer's prompt. `output_file` auto-saves — no extra code needed.

### `crew.py` — The Orchestrator

**Key concept: Sequential process coordinates agent output chaining**

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # tasks run in order
    verbose=True,
)
result = crew.kickoff()
```

`kickoff()` starts execution and blocks until all tasks complete. With `output_file` on the write task, the Markdown report is saved to disk automatically.

### `main.py` — The Entry Point

**Key concept: How `uv run` scripts work**

In `pyproject.toml`:
```toml
[project.scripts]
ai-news-crew = "ai_news_crew.main:run"
```

`uv run ai-news-crew` calls `run()` in `main.py`, which loads env → validates → runs the crew → previews output.

---

## 9. How It Works — The ReAct Loop

Here's a real example of the research agent's reasoning:

```
Iteration 1:
  THOUGHT: I need to find the latest AI news. Start with a broad query.
  ACTION:  Search("latest AI LLM news October 2026")
  OBSERVE: [10 search results: titles, URLs, snippets]

Iteration 2:
  THOUGHT: The Anthropic story is significant. Read the full article.
  ACTION:  Scrape("https://anthropic.com/blog/...")
  OBSERVE: [full article text, ~2,500 words]

Iteration 3:
  THOUGHT: I need more variety. Search for model releases specifically.
  ACTION:  Search("new AI model released October 2026")
  OBSERVE: [10 more results]

... [continues until 8-10 well-sourced stories are found] ...

Final:
  THOUGHT: I have 10 stories with headlines, URLs, dates and summaries
           covering all required categories. That's sufficient.
  FINAL ANSWER: [structured research report]
```

The agent decides for itself when it has enough information. This is the essence of agentic AI — the model controls its own process, not just its output.

---

## 10. Sample Output

The generated `output/ai_news_summary.md` will look something like:

```markdown
# 🤖 AI & LLM News Digest — 2026-10-28

## 📋 Executive Summary

This week in AI was dominated by model releases. **Anthropic** unveiled
Claude 4, while **Google DeepMind** published a landmark research paper...

## 🔥 Top Stories

### Anthropic Releases Claude 4 with 2M Token Context Window

**Anthropic** announced **Claude 4** featuring an unprecedented 2 million
token context window. The model shows particular improvements in code
generation and long-document analysis.

([Anthropic Blog](https://anthropic.com/blog/claude-4))

## 📦 Model Releases & Updates

- **GPT-5 Mini** — OpenAI released a smaller, faster variant...
- **Llama 4** — Meta open-sourced their fourth generation model...

## 📈 Trends to Watch

- **Longer context windows** are the primary competitive differentiator
- **Multimodal models** now handle video in addition to images and audio
- **Open-source** is closing the gap with commercial frontier models

## 🔗 Sources

1. [Anthropic Blog — Claude 4](https://anthropic.com/blog/claude-4)
2. [OpenAI Blog — GPT-5 Mini](https://openai.com/blog/gpt-5-mini)

*Report generated by AI News Crew on October 28, 2026*
```

---

## 11. Troubleshooting

### Authentication errors

```
Error: AuthenticationError / No API key provided
```
Check `.env` exists with real values: `cat .env`

### `AccessDeniedException` from AWS

```
Error: AccessDeniedException: not authorized to perform bedrock:InvokeModel
```
1. Verify the IAM policy includes `bedrock:InvokeModel`
2. Go to AWS Console → Bedrock → **Model access** → enable your model
3. Ensure `AWS_DEFAULT_REGION` matches the region where access was granted

### Model not found

```
Error: The requested model does not exist
```
Check the exact model ID at [AWS Bedrock model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html). IDs change between model versions.

### Serper.dev 401 Unauthorized

API key is invalid or monthly quota is exhausted. Check your [Serper.dev dashboard](https://serper.dev/api-key).

### Agent hits max iterations

The agent reached `max_iter` and stopped. It still produces the best answer it found. Increase `max_iter` in `agents.py` to allow deeper research.

### `temperature` parameter error

Some Amazon Nova models don't accept `temperature`. Comment out the `temperature=` line in `get_llm()` inside `config.py`.

---

## 12. Extending the Project

### Add a Third Agent — Fact Checker

```python
fact_checker = Agent(
    role="AI News Fact Checker",
    goal="Verify claims in the draft against original source URLs",
    tools=[scrape_tool],
    llm=llm,
)
```

### Add an RSS Feed Tool

```python
from crewai import tool
import feedparser

@tool("Read RSS Feed")
def read_rss(feed_url: str) -> str:
    """Reads an RSS feed and returns recent headlines with links."""
    feed = feedparser.parse(feed_url)
    return "\n".join(
        f"- [{e.title}]({e.link}) — {e.get('published', 'No date')}"
        for e in feed.entries[:10]
    )
```

### Schedule Daily Runs

```bash
# Run every day at 8am via cron
0 8 * * * cd /path/to/project && \
  OUTPUT_FILE="output/digest-$(date +\%Y-\%m-\%d).md" uv run ai-news-crew
```

### Use Different LLMs per Agent

```python
# Cheap model for research (many small calls)
researcher_llm = LLM(model="bedrock/amazon.nova-lite-v1:0")

# Powerful model for writing (one large synthesis call)
writer_llm = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
```

### Post Results to Slack

```python
@tool("Post to Slack")
def post_to_slack(message: str) -> str:
    """Posts a message to a configured Slack webhook."""
    import requests, os
    requests.post(os.environ["SLACK_WEBHOOK_URL"], json={"text": message})
    return "Posted successfully"
```

### Enable Agent Memory

```python
# Agents remember information across multiple crew runs
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    memory=True,
)
```

---

## 13. Key Concepts Glossary

| Term | Definition |
|---|---|
| **Agentic AI** | AI that autonomously plans and executes multi-step tasks |
| **ReAct** | Reasoning + Acting — think → act → observe loop used by agents |
| **Agent** | An autonomous AI worker with role, goal, backstory, tools, and LLM |
| **Task** | A work item with description, expected output, and assigned agent |
| **Tool** | A callable function agents use to interact with external systems |
| **Crew** | A team of agents working through a defined process |
| **Sequential Process** | Tasks run in order; prior outputs chain via `context=[]` |
| **Hierarchical Process** | A manager LLM dynamically orchestrates agents |
| **LiteLLM** | Universal LLM adapter (bundled with CrewAI) for 100+ providers |
| **AWS Bedrock** | AWS managed service for foundation models (Claude, Nova, etc.) |
| **boto3** | The AWS SDK for Python; authenticates Bedrock calls |
| **SerperDevTool** | CrewAI tool that queries Google via the Serper.dev API |
| **ScrapeWebsiteTool** | CrewAI tool that fetches full text content from a URL |
| **context window** | Max tokens an LLM can process in one call |
| **temperature** | Sampling control: 0.0 = focused, 1.0 = creative |
| **max_iter** | Max ReAct loops before an agent is forced to answer |
| **uv** | Fast Python package and project manager (written in Rust) |
| **pyproject.toml** | Standard Python project config — defines dependencies and scripts |

---

## Resources

- [CrewAI Documentation](https://docs.crewai.com)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
- [LiteLLM Providers List](https://docs.litellm.ai/docs/providers)
- [Serper.dev API Docs](https://serper.dev/api)
- [uv Documentation](https://docs.astral.sh/uv/)
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI)

---

*Built as a learning project. Contributions and questions welcome!*
