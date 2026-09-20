# Publishing the paper and the visuals

## What gets published

Everything is a static file in the repo — no site, no backend:

| Artefact | Path | Built by |
|---|---|---|
| Paper (PDF + LaTeX source) | `docs/vic-rent-ml-paper.{pdf,tex}` | `docs/build/paper.py` |
| Poster (one-page summary) | `docs/figures/poster.png` | `docs/build/explainer.py` |
| Explainer GIF (recursive mechanism) | `docs/figures/explainer.gif` | `docs/build/explainer.py` |
| Share image (1200×627) | `docs/figures/social_card.png` | `docs/build/figures.py` |

The paper is real LaTeX: `docs/build/paper.py` translates `docs/RESEARCH_PAPER.md`
to `vic-rent-ml-paper.tex` and compiles it with the first TeX engine on `PATH` —
[Tectonic](https://tectonic-typesetting.github.io/) (single binary, recommended),
XeLaTeX, LuaLaTeX or pdfLaTeX. Set `TEX_ENGINE` to point at a binary that is not
on `PATH`. Without an engine the `.tex` is still written but the build exits 1.

The README embeds the GIF and links the poster, so both are visible on the repo
front page with no publishing step.

## Rebuild before publishing

```sh
pip install -e ".[docs]"
python docs/build/analysis.py
python docs/build/evaluation_illusions.py
python docs/build/figures.py
python docs/build/explainer.py
python docs/build/paper.py
```

## LinkedIn post copy

Attach `docs/figures/poster.png` or `docs/figures/explainer.gif` (both upload
directly; the GIF animates in the feed), or `docs/figures/social_card.png` for a
flat 1200×627 card, and put the repo link in the body or first comment.

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
> Paper, code, poster and every number: https://github.com/kelvinvalani/vic-rent-ml
>
> Data: Homes Victoria Rental Report (RTBA bond lodgements), CC BY 4.0.

### Shorter variant

> I scored the same rent-forecasting model three ways. Shuffled split: $7.23 average error per week. Temporal split, one step: $12.04. The way it actually runs — eight quarters ahead, eating its own predictions: $23.80.
>
> Only the last one is real. Full bake-off of 26 candidates over 862 suburb series, including the six a "rent stays the same" baseline beat: https://github.com/kelvinvalani/vic-rent-ml
