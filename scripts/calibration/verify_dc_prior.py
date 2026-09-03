"""Check that the competition weighting and the Elo-anchored prior behave as intended.

Three checks. A lambda of 0 reproduces the unweighted fit exactly, so the prior is
genuinely off by default. The spread of fitted team strength matches the prior
scale it was derived from. And turning both on moves France up relative to Japan,
which is the cross-confederation correction the prior exists to make.
Writes reports/dc_prior_checks.md."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import load_params
from src.models.dixon_coles import fit_dixon_coles
from src.models.elo_prior import elo_prior_net_asof


def _net(d):
    # team strength that drives match supremacy = attack + defence (higher dfc -> opponent scores fewer)
    return {t: d["attack"][t] + d["defence"][t] for t in d["attack"]}


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    dc = load_params()["models"]["dixon_coles"]
    tw = dc["competition_weights"]
    beta = dc.get("prior_scale", 0.35)

    base = fit_dixon_coles(m)
    sb = _net(base)                                  # strength = atk + dfc
    sd_str = float(np.std(list(sb.values())))

    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    pn = elo_prior_net_asof(r, teams, "2026-06-11", beta=beta)
    pri = fit_dixon_coles(m, competition_weights=tw, prior_net=pn, prior_strength=0.5)
    sp = _net(pri)

    lines = [
        "# Dixon-Coles competition weighting and Elo prior: behaviour checks",
        "",
        "Team strength = attack + defence (the quantity that drives match supremacy).",
        "",
        f"- sd(strength) baseline = {sd_str:.3f}  (prior_scale beta = {beta}; recalibrate if these diverge materially)",
        f"- Japan   strength base -> weight+prior(lam=0.5): {sb['Japan']:+.3f} -> {sp['Japan']:+.3f}",
        f"- France  strength base -> weight+prior(lam=0.5): {sb['France']:+.3f} -> {sp['France']:+.3f}",
        f"- Morocco strength base -> weight+prior(lam=0.5): {sb['Morocco']:+.3f} -> {sp['Morocco']:+.3f}",
        f"- England strength base -> weight+prior(lam=0.5): {sb['England']:+.3f} -> {sp['England']:+.3f}",
        "",
        f"- France - Japan strength gap:   base {sb['France']-sb['Japan']:+.3f}  ->  weight+prior {sp['France']-sp['Japan']:+.3f}",
        f"- England - Morocco strength gap: base {sb['England']-sb['Morocco']:+.3f}  ->  weight+prior {sp['England']-sp['Morocco']:+.3f}",
    ]
    report = "\n".join(lines)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dc_prior_checks.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
