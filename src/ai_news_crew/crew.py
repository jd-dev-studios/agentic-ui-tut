# =============================================================================
# src/ai_news_crew/crew.py — Crew assembly and execution
# =============================================================================
#
# PURPOSE OF THIS FILE:
#   This is where everything comes together. A "Crew" in CrewAI is the team
#   of agents working together with a defined process to complete a goal.
#
# =============================================================================
#
# WHAT IS A CREW?
#
#   A Crew is the top-level orchestrator that:
#     - Holds a list of AGENTS
#     - Holds a list of TASKS (the work to be done)
#     - Defines a PROCESS (how tasks are executed)
#     - Calls crew.kickoff() to start execution
#
# PROCESSES — two main options:
#
#   Process.sequential (used here):
#     Tasks run one after another in the order they're listed.
#     Each task can access previous tasks' outputs via context=[].
#     Best for: simple pipelines with a clear order of operations.
#
#     Task 1: research_task → runs → produces output
#     Task 2: write_task    → runs → receives task 1 output as context
#
#   Process.hierarchical:
#     A "manager" LLM coordinates agents, deciding who does what.
#     Best for: complex workflows where tasks need dynamic assignment.
#     Requires setting manager_llm on the Crew.
#
# crew.kickoff() return value:
#   Returns a CrewOutput object. The .raw attribute contains the final
#   agent output as a string. Since we use output_file on the write task,
#   the report is also saved to disk automatically.
#
# =============================================================================

import os

from crewai import Crew, Process

from .agents import create_news_researcher, create_news_writer
from .config import get_llm, get_output_path
from .tasks import create_research_task, create_write_task


def build_crew() -> tuple[Crew, str]:
    """
    Assemble the AI News Crew: agents, tasks, and the crew itself.

    Returns:
        tuple: (crew, output_file_path)
            - crew: The assembled Crew ready to be kicked off
            - output_file_path: Where the Markdown report will be saved
    """
    print("\n🔧 Building AI News Crew...")

    # --- Step 1: Configure the LLM ---
    # A single LLM instance is shared by both agents.
    # You could give each agent a different LLM — e.g. a fast cheap model
    # for research and a slow powerful model for writing.
    llm = get_llm()

    # --- Step 2: Resolve output path ---
    # get_output_path() returns the absolute Path and creates the output directory.
    # We keep the absolute path for display, but pass the RELATIVE path to the
    # Task's output_file parameter.
    #
    # WHY: CrewAI's Task.output_file resolves paths relative to the working
    # directory. If you pass an absolute path (e.g. /home/user/project/output/...)
    # CrewAI strips the leading slash and treats it as relative, which creates
    # a mirrored nested directory structure inside the project root.
    # Passing a plain relative path (e.g. "output/ai_news_summary.md") avoids this.
    output_path = get_output_path()
    output_file_relative = os.environ.get("OUTPUT_FILE", "output/ai_news_summary.md")
    print(f"📄 Report will be saved to: {output_path}")

    # --- Step 3: Create agents ---
    # Each agent gets the shared LLM. Both log their reasoning with verbose=True.
    print("👤 Creating agents...")
    researcher = create_news_researcher(llm)
    writer = create_news_writer(llm)

    # --- Step 4: Create tasks ---
    # Tasks are created with references to their assigned agents.
    # The write_task receives research_task as context (output chaining).
    print("📋 Creating tasks...")
    research_task = create_research_task(researcher)
    # Pass the relative path to the Task — CrewAI resolves it from the cwd
    write_task = create_write_task(writer, research_task, output_file_relative)

    # --- Step 5: Assemble the Crew ---
    crew = Crew(
        # The order of agents matters for hierarchical processes.
        # For sequential, it's informational only.
        agents=[researcher, writer],

        # Tasks run in this order for Process.sequential.
        # Each task's output feeds the next via context=[].
        tasks=[research_task, write_task],

        # Sequential process: task 1 → task 2 → done.
        # Simple, predictable, and easy to understand for a tutorial.
        process=Process.sequential,

        # verbose=True at the Crew level enables CrewAI's internal logging
        # in addition to each agent's individual verbose output.
        verbose=True,
    )

    print("✅ Crew assembled and ready!\n")
    # Return the absolute path string for display/preview, and the relative
    # path that was passed to the Task (for reference)
    return crew, str(output_path)


def run_crew() -> str:
    """
    Build and run the AI News Crew end-to-end.

    Returns:
        str: Path to the generated Markdown report.

    Raises:
        Exception: Re-raises any exception from the crew for handling upstream.
    """
    crew, output_file = build_crew()

    print("🚀 Starting crew execution...\n")
    print("=" * 70)
    print("  The crew is now running. You will see each agent's reasoning")
    print("  process printed below in real time. This is the core of")
    print("  agentic AI — watch how agents think, search, and decide!")
    print("=" * 70 + "\n")

    # crew.kickoff() starts execution.
    # For sequential processes, it runs each task in order and returns
    # a CrewOutput object when all tasks are complete.
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("✅ Crew execution complete!")
    print(f"📄 Report saved to: {output_file}")
    print("=" * 70 + "\n")

    return output_file
