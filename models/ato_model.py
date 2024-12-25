import json

def save_ato_model(config, filepath="models/ato_model_config.json"):
    with open(filepath, "w") as f:
        json.dump(config, f)

def load_ato_model(filepath="models/ato_model_config.json"):
    with open(filepath, "r") as f:
        return json.load(f)