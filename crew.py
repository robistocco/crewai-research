import json
import warnings

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Save a reference to the original showwarning function.
original_showwarning = warnings.showwarning
# Define a custom filter function.
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    # Filter out pydantic warnings: CrewAI tools use @validator instead of @field_validator.
    # Also support for class-based `config` is deprecated, use ConfigDict instead.
    if '/pydantic/_internal/_config.py' in filename or 'crewai_tools/tools' in filename:
        return
    # Show all other warnings
    original_showwarning(message, category, filename, lineno, file, line)
# Replace the warning display function.
warnings.showwarning = custom_showwarning

from crewai_tools import SerperDevTool, WebsiteSearchTool

# Instantiate different models.
file_path = 'gcp_key.json'
with open(file_path, 'r') as file:
    vertex_credentials = json.load(file)
vertex_credentials_json = json.dumps(vertex_credentials)
gemini_llm = LLM(
  model="gemini/gemini-2.0-flash-exp",
  temperature=0.7,
  vertex_credentials=vertex_credentials_json
)

llama_groq_llm = LLM(
  model="groq/llama-3.3-70b-versatile",
  temperature=0.7
)

# Instantiate tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create knowledge sources
content = '''
The users is a software engineer in Google.
The user has several years of experience and a solid understanding of web tech and software engineering principles.'''
string_source = StringKnowledgeSource(
  content=content,
)
text_source = TextFileKnowledgeSource(
  file_paths=["user_preferences.txt"]
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class ResearchCrew():
  """Roberto crew"""

  # Learn more about YAML configuration files here:
  # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
  # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
  agents_config = 'config/agents.yaml'
  tasks_config = 'config/tasks.yaml'

  # If you would like to add tools to your agents, you can learn more about it here:
  # https://docs.crewai.com/concepts/agents#agent-tools
  @agent
  def researcher(self) -> Agent:
    return Agent(
      config=self.agents_config['researcher'],
      llm=gemini_llm,
      tools=[search_tool, web_rag_tool],
      verbose=True
    )

  @agent
  def reporting_analyst(self) -> Agent:
    return Agent(
      config=self.agents_config['reporting_analyst'],
      llm=llama_groq_llm,
      verbose=True
    )

  # To learn more about structured task outputs,
  # task dependencies, and task callbacks, check out the documentation:
  # https://docs.crewai.com/concepts/tasks#overview-of-a-task
  @task
  def research_task(self) -> Task:
    return Task(
      config=self.tasks_config['research_task'],
    )

  @task
  def reporting_task(self) -> Task:
    return Task(
      config=self.tasks_config['reporting_task'],
      output_file='report.md'
    )

  @crew
  def crew(self) -> Crew:
    """Creates the Roberto crew"""
    # To learn how to add knowledge sources to your crew, check out the documentation:
    # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

    return Crew(
      agents=self.agents, # Automatically created by the @agent decorator
      tasks=self.tasks, # Automatically created by the @task decorator
      process=Process.sequential,
      verbose=True,
      planning=True,
      knowledge_sources=[string_source, text_source],
      # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
    )
