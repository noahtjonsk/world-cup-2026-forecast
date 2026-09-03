import yaml

def load_params(path="config/params.yaml"):
    """Load the YAML params file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
