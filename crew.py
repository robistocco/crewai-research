""" Crew. """
import json
import warnings

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Save a reference to the original 'showwarning' function.
original_showwarning = warnings.showwarning
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

from crewai_tools import SerperDevTool, WebsiteSearchTool  # pylint: disable=C0413

# Instantiate a Google Gemini model.
FILE_PATH = 'gcp_key.json'
with open(FILE_PATH, 'r', encoding="utf-8") as json_file:
  vertex_credentials = json.load(json_file)
vertex_credentials_json = json.dumps(vertex_credentials)
gemini_llm = LLM(
  model="gemini/gemini-1.5-pro",
  temperature=0.7,
  vertex_credentials=vertex_credentials_json,
    max_tokens=6000  # Limit the output size.
)

# Instantiate a DeepSeek R1 model, hosted on Groq.
groq_llm = LLM(
  model="groq/deepseek-r1-distill-llama-70b",
  temperature=0.7
)

# Instantiate tools.
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create knowledge sources.
CONTENT = '''
The users is a software engineer in Google.
The user has several years of experience and a solid understanding of web tech
and software engineering principles.'''
string_source = StringKnowledgeSource(
  content=CONTENT,
)
text_source = TextFileKnowledgeSource(
  file_paths=["user_preferences.txt"]
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators.
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ResearchCrew():
  """Agents crew"""

  # Learn more about YAML configuration files here:
  # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
  # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'

  # If you would like to add tools to your agents, you can learn more about it here:
  # https://docs.crewai.com/concepts/agents#agent-tools
  @agent
  def researcher(self) -> Agent:
    """ Instantiate the researcher agent. """
    return Agent(
      config=self.agents_config['researcher'],  # pylint:disable=E1126
      llm=groq_llm,
      tools=[search_tool, web_rag_tool],
      verbose=True
    )

  @agent
  def reporting_analyst(self) -> Agent:
    """ Instantiate the reporting analyst agent. """
    return Agent(
      config=self.agents_config['reporting_analyst'],  # pylint:disable=E1126
      verbose=True
    )

  # To learn more about structured task outputs,
  # task dependencies, and task callbacks, check out the documentation:
  # https://docs.crewai.com/concepts/tasks#overview-of-a-task
  @task
  def research_task(self) -> Task:
    """ Instantiate the research task. """
    return Task(
      config=self.tasks_config['research_task'],  # pylint:disable=E1126
    )

  @task
  def reporting_task(self) -> Task:
    """ Instantiate the reporting task. """
    return Task(
      config=self.tasks_config['reporting_task'],  # pylint:disable=E1126
      output_file='report.md'
    )

  @crew
  def crew(self) -> Crew:
    """Creates the crew."""
    # To learn how to add knowledge sources to your crew, check out the documentation:
    # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

    # self.agents and self.tasks are automatically created by the @agent and
    # @task decorator.
    return Crew(
      agents=self.agents,  # pylint:disable=E1101
      tasks=self.tasks,  # pylint:disable=E1101
      process=Process.sequential,
      verbose=True,
      planning=True,
      planning_llm=gemini_llm,
      knowledge_sources=[string_source, text_source]
    )
