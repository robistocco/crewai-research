""" Main. """
# Enable for Langtrace.
# from dotenv import load_dotenv
# load_dotenv()
import warnings
from datetime import datetime
from textwrap import dedent

from crew import ResearchCrew

# Must precede any llm module imports.
# from langtrace_python_sdk import langtrace
# langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))


# Save a reference to the original 'showwarning' function.
original_showwarning = warnings.showwarning
# Define a custom filter function.


# To avoid Pydantic deprecation warnings, add before importing CrewAI tools:
def custom_showwarning(message, category, filename, lineno, file=None, line=None):  # pylint: disable=R0913,R0917
  """ Filter out pydantic warnings.
  CrewAI depends on Pydantic v2 but the "tools" package uses v1 notations that
  have been deprecated, and that triggers many warnings output in stdout.
  We want to filter them out.
  """
  if '/pydantic/_internal/' in filename or 'crewai_tools/tools' in filename:
    return
  # Show all other warnings.
  original_showwarning(message, category, filename, lineno, file, line)


# Replace the warning display function.
warnings.showwarning = custom_showwarning


if __name__ == "__main__":
  topic = input(dedent(""">>> Topic to research: """))

  inputs = {
    "topic": topic,
    "current_year": str(datetime.now().year)
  }

  try:
    ResearchCrew().crew().kickoff(inputs=inputs)
  except Exception as e:
    # pylint: disable=W0719
    raise Exception(f"An error occurred while running the crew: {e}") from e
    # pylint: enable=W0719
