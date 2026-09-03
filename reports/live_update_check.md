# Live-update engine: end-to-end check

- COARSE note carried: 'Live updates are COARSE: Elo + lineups + summary stats only, not event-level. Forecasts refresh per result; the CatBoost W/D/L model is reused (periodic out-of-band refit), while the Dixon-Coles goal model and the Monte-Carlo simulation are rerun each cycle.'
- upcoming fixtures forecast: 72
- W/D/L row sums in [0.999, 1.001]: True
- expected goals range: home [0.44, 2.65], away [0.44, 2.32]
- IDEMPOTENT (ratings.parquet md5 identical on re-run): True  (e0596223fc87882c7bd3c72bbbed9a8d == e0596223fc87882c7bd3c72bbbed9a8d)
- first cycle wall time: 2150.6s

## match_predictions (head 10)

```
    home_team              away_team   p_home   p_draw   p_away  exp_goals_home  exp_goals_away
       Mexico           South Africa 0.876850 0.087643 0.035507        1.854629        0.612001
  South Korea         Czech Republic 0.463432 0.286361 0.250207        1.734679        0.886187
       Canada                  Qatar 0.815426 0.127480 0.057094        2.417109        0.748495
  Switzerland Bosnia and Herzegovina 0.760029 0.160892 0.079079        2.146574        0.593298
       Brazil                Morocco 0.321345 0.298979 0.379676        0.856328        0.810412
        Haiti               Scotland 0.087404 0.172004 0.740592        1.187941        1.426601
United States              Australia 0.276356 0.292844 0.430800        0.911799        1.317040
       Turkey               Paraguay 0.450569 0.289138 0.260292        0.965737        1.164345
      Germany            Ivory Coast 0.731891 0.176870 0.091240        1.320020        0.853884
      Ecuador                 Panama 0.630599 0.228159 0.141242        1.192514        0.618256
```

## title odds (top 8)

```
tournament      team round   prob   ci_low  ci_high
      2026 Argentina     W 0.1221 0.115697 0.128802
      2026     Spain     W 0.1209 0.114998 0.127200
      2026    Brazil     W 0.0829 0.077295 0.088400
      2026     Japan     W 0.0638 0.059200 0.068100
      2026   Morocco     W 0.0527 0.048500 0.057302
      2026   England     W 0.0493 0.045200 0.053302
      2026    France     W 0.0469 0.042700 0.051000
      2026  Colombia     W 0.0459 0.041900 0.049902
```