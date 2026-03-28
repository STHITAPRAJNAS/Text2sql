"""
ADK App Entry Point — text2sql_prism
=====================================
Google ADK requires each app directory to expose `root_agent`.
The get_fast_api_app() loader discovers this file via the agents_dir convention.

  agents/
    text2sql_prism/      ← app_name used in all ADK API paths
      agent.py           ← this file, must export root_agent
      __init__.py

The root_agent here is the full PRISM orchestrator — the top-level
SequentialAgent that runs P→R→I→S→M phases.
"""
from agents.orchestrator import get_orchestrator

# ADK's get_fast_api_app loader looks for this name specifically
root_agent = get_orchestrator()
