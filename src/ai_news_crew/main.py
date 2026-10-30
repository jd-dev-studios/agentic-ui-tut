# =============================================================================
# src/ai_news_crew/main.py — CLI entry point
# =============================================================================
#
# PURPOSE OF THIS FILE:
#   The application's entry point. This is what runs when you execute:
#       uv run ai-news-crew
#
#   It handles:
#     1. Loading and validating environment variables
#     2. Printing a welcome banner
#     3. Delegating execution to crew.py
#     4. Displaying the final output location and a preview
#
# =============================================================================
#
# HOW THE ENTRY POINT WORKS:
#
#   In pyproject.toml we define:
#       [project.scripts]
#       ai-news-crew = "ai_news_crew.main:run"
#
#   This tells uv/pip: when someone runs `ai-news-crew`, call the `run()`
#   function in the `ai_news_crew.main` module.
#
#   The `if __name__ == "__main__":` block at the bottom also lets you run
#   this file directly with: uv run python src/ai_news_crew/main.py
#
# =============================================================================

import sys
from datetime import datetime
from pathlib import Path

from .config import load_environment, validate_environment
from .crew import run_crew


def print_banner() -> None:
    """Print a welcome banner with basic info about the run."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 70)
    print("  🤖  AI News Crew — Powered by CrewAI + AWS Bedrock")
    print("=" * 70)
    print(f"  Run started: {now}")
    print(f"  What this does:")
    print(f"    1. 🔍 Research Agent searches for the latest AI/LLM news")
    print(f"    2. ✍️  Writer Agent creates a structured Markdown summary")
    print(f"    3. 💾 Report is saved to the output/ directory")
    print("=" * 70 + "\n")


def print_preview(output_file: str) -> None:
    """
    Read and print the first 30 lines of the generated report as a preview.

    Args:
        output_file: Path to the generated Markdown file.
    """
    path = Path(output_file)
    if not path.exists():
        print("⚠️  Output file not found — the agent may not have written it yet.")
        return

    print("\n" + "=" * 70)
    print("  📄 REPORT PREVIEW (first 30 lines)")
    print("=" * 70)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Print up to 30 lines as a preview
    preview_lines = lines[:30]
    print("".join(preview_lines))

    if len(lines) > 30:
        print(f"  ... [{len(lines) - 30} more lines in the full report]")

    print("\n" + "=" * 70)
    print(f"  Full report: {path.absolute()}")
    print("=" * 70 + "\n")


def run() -> None:
    """
    Main entry point for the AI News Crew application.

    Execution flow:
        1. Load .env file into environment variables
        2. Validate all required variables are set
        3. Print welcome banner
        4. Build and run the crew
        5. Preview the generated report
    """
    # Step 1: Load environment variables from .env file
    # This must happen before validate_environment() or get_llm()
    load_environment()

    # Step 2: Validate required env vars — exits with helpful message if missing
    validate_environment()

    # Step 3: Welcome banner
    print_banner()

    try:
        # Step 4: Build agents, tasks, crew — then kickoff!
        # This is where the magic happens. Watch the terminal output to see
        # each agent's reasoning process in real time.
        output_file = run_crew()

        # Step 5: Preview the generated report
        print_preview(output_file)

        print("🎉 All done! Your AI news digest is ready.\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Run interrupted by user (Ctrl+C). Exiting.")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ An error occurred during crew execution:\n   {type(e).__name__}: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Check your AWS credentials and region in .env")
        print("  2. Verify the model is enabled in AWS Bedrock console")
        print("  3. Check your SERPER_API_KEY is valid at serper.dev")
        print("  4. Run with verbose=True in crew.py to see detailed logs")
        print("  5. See README.md for detailed troubleshooting steps\n")
        raise


# Allow running directly with: uv run python src/ai_news_crew/main.py
if __name__ == "__main__":
    run()
