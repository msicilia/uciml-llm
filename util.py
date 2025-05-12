from smolagents import CodeAgent, LiteLLMModel, tool, AgentLogger
import pandas as pd
from rich.console import Console

def build_prompt(prompt_template: str, config: dict):
    """Build a prompt from a template file, a problem statement and a configuration dictionary."""
    with open(prompt_template, "r") as f:
        prompt = f.read()
    return prompt.format( **config)

@tool
def read_dataset(dataset_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the required dataset from the given path and returns two dataframes: features and targets.
    Args:
        dataset_path (str): Path to the dataset folder.
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the features and targets dataframes."""
    features = pd.read_csv(f"{dataset_path}/features.csv")
    targets = pd.read_csv(f"{dataset_path}/targets.csv")
    return features, targets



class Model:
    """Simple class to wrap the LiteLLMModel and CodeAgent classes."""
    def __init__(self, model_name, config: dict, temp: float):
        self.model = LiteLLMModel(model_name, temp=temp)
        self.temp = temp
        self.config = config
        self.system_prompt = "You are an experienced data scientist.\n"
        self.agent = CodeAgent(tools=[read_dataset], 
                               model=self.model, 
                               additional_authorized_imports=['numpy', 'pandas', 'sklearn.*'],
                               max_steps=3,
                               logger = AgentLogger(console=Console(record=True)))
      
    def run(self):
        """Buld the prompt for the given problem statement and run the model."""
        self.prompt = build_prompt(self.config["prompt_template"], self.config)
        print(self.prompt)
        return self.agent.run(self.system_prompt + " \n" + self.prompt)