"""PFF FC deep-tier loaders. kloppy.pff verified via kloppy docs;
Converts PFF/kloppy event data through socceraction.spadl.kloppy.

NOTE: the raw PFF release ships metadata as CSVs (metadata.csv, players.csv,
rosters.csv, competitions.csv) but kloppy's pff.load_tracking expects per-game
{game_id}.json for meta_data/players_meta_data, convert the CSVs to per-game
JSON first, or use the loader's expected layout."""

def load_pff_tracking(meta_data, players_meta_data, raw_data, coordinates="pff"):
    from kloppy import pff
    return pff.load_tracking(
        meta_data=meta_data,
        players_meta_data=players_meta_data,
        raw_data=raw_data,
        coordinates=coordinates,
    )

def pff_events_to_spadl(dataset, game_id):
    """Convert a kloppy-loaded PFF event dataset to SPADL actions."""
    from socceraction.spadl.kloppy import convert_to_actions
    return convert_to_actions(dataset, game_id=game_id)
