from src.config import load_params

def load_tournament_format(year, path="config/tournaments.yaml"):
    """Load one tournament's format dict (group shape, rounds, hosts, knockout
    seeding) from YAML. Reuses the params loader; `year` is the top-level key."""
    return load_params(path)[str(year)]
