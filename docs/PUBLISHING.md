# Publishing the paper and the interactive page

## What gets published

| Artefact | Path | Public URL once Pages is on |
|---|---|---|
| Interactive explainer | `docs/index.html` | `https://kelvinvalani.github.io/vic-rent-ml/` |
| Paper | `docs/vic-rent-ml-paper.pdf` | `https://kelvinvalani.github.io/vic-rent-ml/vic-rent-ml-paper.pdf` |
| Share image (1200×627) | `docs/figures/social_card.png` | `https://kelvinvalani.github.io/vic-rent-ml/figures/social_card.png` |

Both pages are static and self-contained: the interactive page embeds its own data, SVG charts, CSS, and JavaScript, so there is no backend, no chart library, and no tracking.

## Turn on GitHub Pages

`.github/workflows/pages.yml` deploys `docs/` on every push to `main`. It needs Pages enabled once:

**Settings → Pages → Build and deployment → Source: GitHub Actions.**

Then re-run the workflow (Actions → *pages* → *Run workflow*) and the site is live at the URL above.

## Rebuild before publishing

```sh
pip install -e ".[docs]"
python docs/build/analysis.py
python docs/build/evaluation_illusions.py
python docs/build/figures.py
python docs/build/paper.py
python docs/build/interactive.py
```

The PDF step drives headless Chrome, so a Chrome or Chromium binary must be on `PATH`.

## LinkedIn post copy

Attach `docs/figures/social_card.png` (LinkedIn previews link images poorly; uploading the PNG directly gives a much larger card) and put the link in the first comment or in the body.

> Most rent-prediction demos report an R² above 0.99. Mine does too — and that number is worthless.
>
> I rebuilt a Victorian rental forecaster around one question: what error do you get when the model runs the way it actually ships?
>
> Same model, same features, three ways of scoring it:
> • Shuffled 80/20 split, one step ahead: $7.23 MAE per week (R² 0.993)
> • Temporal split, one step ahead, real lag supplied: $12.04
> • Temporal split, eight quarters ahead, the model eating its own forecasts: $23.80
>
> The first number is the one that gets posted. The last one is the only one that describes the product — and it is three times worse.
>
> What else came out of scoring 26 candidates honestly across 862 suburb-group rent series:
> • "Rent stays the same" beats six of the 24 learned configurations, including four gradient-boosting variants. Baselines were allowed to win.
> • A Huber regressor on the quarterly *change* in the median won: $23.80 vs $25.98 for persistence on validation, $14.87 vs $16.85 on a frozen 2025 audit the model never saw before selection.
> • That edge holds under a bootstrap that resamples whole suburb groups, not rows — rows inside a suburb are not independent.
> • Only three of the eight forecast horizons have real outcomes yet. The other five ship with calibrated but unaudited bands, stated rather than hidden.
> • It forecasts group medians, not your apartment.
>
> Interactive walkthrough (no login, no backend): <PAGES URL>
> Paper, code, and every number: https://github.com/kelvinvalani/vic-rent-ml
>
> Data: Homes Victoria Rental Report (RTBA bond lodgements), CC BY 4.0.

Replace `<PAGES URL>` after Pages is live.

### Shorter variant

> I scored the same rent-forecasting model three ways. Shuffled split: $7.23 average error per week. Temporal split, one step: $12.04. The way it actually runs — eight quarters ahead, eating its own predictions: $23.80.
>
> Only the last one is real. Here is the full bake-off of 26 candidates over 862 suburb series, including the six a "rent stays the same" baseline beat: <PAGES URL>
