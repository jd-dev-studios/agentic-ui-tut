# =============================================================================
# src/ai_news_crew/config.py — LLM configuration and environment setup
# =============================================================================
#
# PURPOSE OF THIS FILE:
#   This module handles all configuration for the application:
#     1. Loading environment variables from the .env file
#     2. Validating that required variables are present
#     3. Constructing the LLM (Large Language Model) object for AWS Bedrock
#
# =============================================================================
#
# HOW CREWAI CONNECTS TO AWS BEDROCK:
#
#   CrewAI ships with LiteLLM — a universal LLM adapter library that provides
#   a single interface to 100+ LLM providers. You don't call the Bedrock API
#   directly; instead you create a `crewai.LLM` object with a model string,
#   and LiteLLM handles the translation.
#
#   The model string format for Bedrock is:
#       "bedrock/<aws-model-id>"
#   e.g. "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
#
#   LiteLLM then uses boto3 under the hood to make the actual API call to
#   AWS Bedrock. boto3 finds your AWS credentials using the standard chain:
#
#   Credential Resolution Order (boto3 checks these in order):
#     1. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
#     2. AWS Config file: ~/.aws/credentials  (set via `aws configure`)
#     3. IAM Instance Profile (when running on EC2/ECS/Lambda)
#
#   This means: on a developer machine use env vars or ~/.aws/credentials,
#   in production on EC2/ECS attach an IAM role — boto3 picks it up auto.
#
# =============================================================================

import os
import sys
from pathlib import Path

# python-dotenv reads a .env file and loads each line as an environment variable.
# override=False means existing system env vars take precedence over .env values.
from dotenv import load_dotenv

# crewai.LLM is CrewAI's wrapper around LiteLLM.
from crewai import LLM


def load_environment() -> None:
    """
    Load environment variables from the .env file.

    Looks for .env in the project root (three levels up from this file:
    src/ai_news_crew/config.py → src/ai_news_crew → src → project_root).
    """
    # __file__ = /path/to/project/src/ai_news_crew/config.py
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file, override=False)
        print(f"✅ Loaded environment from: {env_file}")
    else:
        # .env doesn't exist — the user may be relying on system env vars
        # (valid in CI/CD). We warn the user and continue.
        print(
            f"⚠️  No .env file found at {env_file}\n"
            f"   Relying on system environment variables.\n"
            f"   Copy .env.example to .env to get started locally."
        )
        load_dotenv()  # No-op if no file found, but sets up dotenv internals


def validate_environment() -> None:
    """
    Check that all required environment variables are set.
    Exits with a helpful message if any are missing.
    """
    required_vars = {
        "AWS_DEFAULT_REGION": (
            "AWS region for Bedrock (e.g. us-east-1). "
            "See .env.example for supported regions."
        ),
        "BEDROCK_MODEL_ID": (
            "Bedrock model string (e.g. bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0). "
            "See .env.example for all options."
        ),
        "SERPER_API_KEY": (
            "Serper.dev API key for web search. "
            "Get a free key at https://serper.dev (2,500 free queries/month)."
        ),
    }

    missing = []
    for var, description in required_vars.items():
        if not os.environ.get(var):
            missing.append(f"  ❌ {var}\n     → {description}")

    if missing:
        print("\n🚨 Missing required environment variables:\n")
        print("\n".join(missing))
        print(
            "\n💡 Quick fix:\n"
            "   cp .env.example .env\n"
            "   # Then edit .env with your real values\n"
        )
        sys.exit(1)

    print("✅ All required environment variables are set.")


def get_llm() -> LLM:
    """
    Construct and return a configured CrewAI LLM object for AWS Bedrock.

    Returns:
        LLM: A CrewAI LLM instance pointing at the configured Bedrock model.

    How this works:
        - The `model` string "bedrock/anthropic.claude-..." tells LiteLLM
          to route this request through AWS Bedrock.
        - LiteLLM uses boto3 internally to sign and send authenticated requests.
        - boto3 finds your AWS credentials from environment variables automatically.
        - `temperature` controls creativity (0.0 = deterministic, 1.0 = creative).
          0.3-0.7 works well for research and summarisation.
        - `max_tokens` caps the response length. 4096 is generous for a summary.

    Note on temperature:
        Some Bedrock models (e.g. Amazon Nova) may not support `temperature`.
        If you see an error about an unsupported parameter, remove it from
        the LLM() call below and check your model's documentation.
    """
    model_id = os.environ.get(
        "BEDROCK_MODEL_ID",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    temperature = float(os.environ.get("BEDROCK_TEMPERATURE", "0.7"))

    print(f"🤖 Configuring LLM: {model_id} (temperature={temperature})")

    # crewai.LLM wraps LiteLLM. The `model` string prefix "bedrock/" tells
    # LiteLLM which provider adapter to use. Everything after the slash is
    # passed to boto3 as the Bedrock Model ID.
    llm = LLM(
        model=model_id,
        temperature=temperature,
        max_tokens=4096,
    )

    return llm


def get_output_path() -> Path:
    """
    Return the resolved absolute path for the output Markdown file.
    Creates the output directory if it doesn't already exist.

    Returns:
        Path: Absolute path to the output .md file.
    """
    project_root = Path(__file__).parent.parent.parent
    output_file = os.environ.get("OUTPUT_FILE", "output/ai_news_summary.md")

    # Resolve relative paths against the project root so the file always
    # lands in the right place regardless of where the command is run from.
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path
