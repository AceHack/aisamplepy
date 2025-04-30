import os
import json
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Use getenv to avoid errors if AZURE_OPENAI_ENDPOINT is not set, though it should be.
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY", "")
# Setting AZURE_API_BASE might also be helpful for some LiteLLM versions
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
# Ensure the deployment name is set for CrewAI/LiteLLM to pick up
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

# Now import CrewAI and Langchain components
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew # Import crew decorator
from langchain_openai import AzureChatOpenAI # Re-import AzureChatOpenAI

# Import the custom tool from the tools directory
from tools.image_analysis_tool import ImageAnalysisTool
# Import the models from the models file
from models import BusinessOpportunity, OpportunitiesReport, StructuredImageAnalysisInput, Finding # Added StructuredImageAnalysisInput, Finding

# Configure the Azure LLM (re-introduced)
azure_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # Load model name from env var, default to empty string if not set
    model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", "")
)

# Instantiate the custom tool (remains the same)
image_analyzer = ImageAnalysisTool()

# Define the CrewBase class to load configurations
@CrewBase
class OpportunityCrew:
    """OpportunityCrew leveraging YAML configuration."""
    agents_config = 'agents.yaml'
    tasks_config = 'tasks.yaml'

    # Inject the tools and manually created LLM into the loaded agent
    @agent
    def opportunity_analyst(self) -> Agent:
        ag = Agent(config=self.agents_config['opportunity_analyst'],
                   llm=azure_llm, # Pass the manually created LLM object
                   tools=[image_analyzer],
                   verbose=True)
        return ag

    # Task definition remains largely the same, referencing the agent method
    @task
    def analysis_task(self) -> Task:
        # Note: The output_pydantic is set in the run method for dynamic task creation
        # The description will also be formatted/set dynamically in run
        return Task(config=self.tasks_config['analysis_task'],
                    agent=self.opportunity_analyst(),
                    output_pydantic=OpportunitiesReport,
                    verbose=True)

    # Explicitly define the crew using the @crew decorator
    @crew
    def crew(self) -> Crew:
        """Creates the Opportunity crew"""
        return Crew(
            agents=self.agents, # Use self.agents and self.tasks provided by CrewBase
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    def run(self, analysis_input: StructuredImageAnalysisInput):
        """Run the analysis on the provided structured input."""
        # Prepare findings as a JSON string for embedding in the description
        findings_list = analysis_input.model_dump()['findings']
        try:
            findings_json = json.dumps(findings_list, indent=2)
        except TypeError:
            findings_json = str(findings_list) # Fallback if JSON serialization fails

        # Inputs for kickoff (still potentially useful for other context)
        inputs = {
            'formId': analysis_input.formId,
            'findings': findings_list,
        }

        # Get the agent instance defined in the class
        analyst_agent = self.opportunity_analyst()

        # Create the task dynamically within the run method
        analysis_task_data = self.tasks_config['analysis_task']
        # Format the description with the actual data
        description = analysis_task_data['description'].format(
            formId=analysis_input.formId,
            findings_json=findings_json
        )

        analysis_task = Task(
            description=description, # Use the formatted description
            expected_output=analysis_task_data['expected_output'],
            agent=analyst_agent,
            output_pydantic=OpportunitiesReport,
            verbose=True
        )

        # Create the Crew instance for this specific run
        opportunity_crew = Crew(
            agents=[analyst_agent],
            tasks=[analysis_task],
            process=Process.sequential,
            verbose=True
        )

        # Execute the Crew
        print("Starting Crew execution...")
        try:
            # Kick off the crew - inputs might still be useful for broader context
            result = opportunity_crew.kickoff(inputs=inputs)
            print("\\nCrew execution finished.")
            print("\\nFinal Result:")
            print(result)
            return result
        except Exception as e:
            print(f"\\nAn error occurred during crew execution: {e}")
            return None

# This part ensures the script can be run directly
if __name__ == '__main__':
    # Create sample structured input data with real image URLs
    sample_input = StructuredImageAnalysisInput(
        formId="FORM_REAL_IMAGES_TEST", # Updated formId for clarity
        findings=[
            Finding(description="Finding 1",
                    image_url="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*vdvf-ds54uMEZQO7ZpY9iA.png"), # Correct URL
            Finding(description="Finding 2",
                    image_url="https://www.networkbachelor.com/wp-content/uploads/2021/01/Azure.png"), # Real microservices diagram
        ]
    )

    # Instantiate the crew runner
    crew_runner = OpportunityCrew()
    # Run the crew with the sample structured input
    crew_runner.run(sample_input)
