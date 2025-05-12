import pandas as pd
import os
from util import build_prompt, Model
from itertools import product
import json
from rich.markdown import Markdown

subfolders = [f for f in os.listdir('./datasets')]
    
for folder in subfolders:
    model_names = [ "ollama/qwen2.5-coder",   
                    #"ollama/yi-coder",
                    #"ollama/codellama",
                    # "ollama/codegemma", 
                    #"ollama/MFDoom/deepseek-coder-v2-tool-calling",
                    # "ollama/codestral"
                    ]
    temperatures = [0.25, 0.5]
    metadata = json.loads(open(f"datasets/{folder}/metadata.json", "r").read())

    feature_desc = pd.read_csv(f"datasets/{folder}/variables.csv")
    config = {}
    config["prompt_template"] = f"./prompt.txt"
    config["feature_desc"] = feature_desc.to_markdown()
    config["path"] = f"./datasets/{folder}/"
    config["abstract"] = metadata["abstract"]
    config["summary"] = metadata["additional_info"]["summary"]

    for model_name, temp in product(model_names, temperatures):
        model = Model(model_name, config, temp) 
        out = model.run()
        filename = f"results/{folder}_dataset_{model_name.split('/')[1]}_{str(temp).replace('.', '_')}.md"
        with open(filename, "w") as f:
            f.write(str(out))
        model.agent.logger.console.save_html(f"{filename}_logs.html")
        # with open(f"{filename}_logs.md", "w") as f:
        #     f.write("\n".join([str(step) for step in model.agent.logger.]))
        



