# Dixon-Coles competition weighting and Elo prior: behaviour checks

Team strength = attack + defence (the quantity that drives match supremacy).

- sd(strength) baseline = 0.831  (prior_scale beta = 0.35; recalibrate if these diverge materially)
- Japan   strength base -> weight+prior(lam=0.5): +1.977 -> +1.767
- France  strength base -> weight+prior(lam=0.5): +1.854 -> +1.659
- Morocco strength base -> weight+prior(lam=0.5): +1.936 -> +1.791
- England strength base -> weight+prior(lam=0.5): +1.866 -> +1.749

- France - Japan strength gap:   base -0.123  ->  weight+prior -0.109
- England - Morocco strength gap: base -0.069  ->  weight+prior -0.042