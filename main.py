# Enable for Langtrace.
# from dotenv import load_dotenv
# load_dotenv()

import sys

from datetime import datetime
from textwrap import dedent

# Must precede any llm module imports.
# from langtrace_python_sdk import langtrace
# langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))

from crew import ResearchCrew

if __name__ == "__main__":
  topic = input(dedent(""">>> Topic to research: """))

  inputs = {
    "topic": topic,
    "current_year": str(datetime.now().year)
  }
  
  try:
    ResearchCrew().crew().kickoff(inputs=inputs)
  except Exception as e:
    raise Exception(f"An error occurred while running the crew: {e}")
