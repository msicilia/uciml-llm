import pandas as pd
import os
from util import build_prompt, Model
from itertools import product
import json

subfolders = [f for f in os.listdir('./datasets')]
    
for folder in subfolders:
    model_names = [ "ollama/qwen2.5-coder",   
                    #"ollama/yi-coder",
                    #"ollama/codellama", "ollama/codegemma", "ollama/codestral"
                    ]
    temperatures = [0.25, 0.5]
    metadata = json.loads(open(f"datasets/{folder}/metadata.json", "r").read())
    config = metadata
    feature_desc = pd.read_csv(f"datasets/{folder}/variables.csv")
    config["prompt_template"] = f"./prompt.txt"
    config["feature_desc"] = feature_desc.to_markdown()
    config["path"] = f"./datasets/{folder}/"
    for model_name, temp in product(model_names, temperatures):
        model = Model(model_name, config, temp) 
        out = model.run()
        with open(f"results/{folder}_dataset_{folder}_{model_name.split('/')[1]}_{str(temp).replace('.', '_')}.md", "w") as f:
                f.write(str(out))



