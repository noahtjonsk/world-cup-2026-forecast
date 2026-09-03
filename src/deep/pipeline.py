"""Thin wrappers around socceraction. Smoke-tested with real
StatsBomb data via the manual command in Step 5, not unit-tested with fixtures."""
import pandas as pd

def events_to_spadl(events, home_team_id):
    import socceraction.spadl as spadl
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
    return spadl.play_left_to_right(spadl.add_names(actions), home_team_id)

def fit_xt(actions_ltr, l=16, w=12):
    import socceraction.xthreat as xthreat
    return xthreat.ExpectedThreat(l=l, w=w).fit(actions_ltr)

def rate_xt(model, actions_ltr):
    import socceraction.xthreat as xthreat
    mov = xthreat.get_successful_move_actions(actions_ltr).copy()
    mov["xt_value"] = model.rate(mov)
    return mov

def compute_vaep_ratings(loader, games):
    """Full VAEP pipeline for a set of games (verified recipe): events -> SPADL ->
    features/labels -> fit -> rate. Returns actions with offensive/defensive/vaep_value."""
    from socceraction.vaep import VAEP
    import socceraction.spadl as spadl
    all_actions, feats, labs = [], [], []
    vaep = VAEP(nb_prev_actions=3)
    for g in games.itertuples():
        ev = loader.events(g.game_id)
        act = spadl.add_names(spadl.statsbomb.convert_to_actions(ev, g.home_team_id))
        gs = pd.Series({"game_id": g.game_id, "home_team_id": g.home_team_id})
        all_actions.append((gs, act))
        feats.append(vaep.compute_features(gs, act))
        labs.append(vaep.compute_labels(gs, act))
    vaep.fit(pd.concat(feats, ignore_index=True), pd.concat(labs, ignore_index=True), learner="catboost")
    rated = [pd.concat([act.reset_index(drop=True), vaep.rate(gs, act).reset_index(drop=True)], axis=1)
             for gs, act in all_actions]
    return pd.concat(rated, ignore_index=True)
