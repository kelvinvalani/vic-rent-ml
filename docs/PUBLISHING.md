# Publishing the paper and the visuals

## What gets published

Everything is a static file in the repo — no site, no backend:

| Artefact | Path | Built by |
|---|---|---|
| Paper (PDF + LaTeX source) | `docs/vic-rent-ml-paper.{pdf,tex}` | `docs/build/paper.py` |
| Poster (one-page summary) | `docs/figures/poster.png` | `docs/build/explainer.py` |
| Explainer GIF (recursive mechanism) | `docs/figures/explainer.gif` | `docs/build/explainer.py` |
| LinkedIn GIF (paper figure) | `docs/figures/linkedin.gif` | `docs/build/explainer.py` |
| LinkedIn card (static square) | `docs/figures/linkedin_card.png` | `docs/build/explainer.py` |
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

Attach `docs/figures/linkedin.gif` as the native image. It is a square paper figure
that autoplays: the Brunswick 2-bed recursive forecast on top, and the same model
scored three ways underneath (shuffled $7, temporal $12, recursive at the current
horizon). Use `docs/figures/linkedin_card.png` if you want a still.

> A two-year rent forecast that prints one dollar is usually lying.
>
> The useful number is how wrong that dollar is allowed to be.
>
> The GIF is Brunswick 2-bedroom flats — Homes Victoria’s official moving-annual median, not a listing. Today that typical rent is $600 a week. Next quarter, 80% of past misses sat within about ±$17 ($587–$622). One year out: about ±$68. Two years out: about ±$116 ($516–$749).
>
> That widening band is not a cop-out. It is the planning tool.
>
> Next quarter is tight enough to budget the next rent period.
> A year out, you can still ask: if typical rents in this suburb group run to the top of the band, does a 12-month lease still fit?
> Two years out you will not nail $633. You can still decide whether the area works if the typical median sits anywhere from the mid-500s to the mid-700s. Compare two suburbs on those ranges, not on two fake-precise dollars. Size a relocation budget off the top of the band, not off a point guess.
>
> I measured this on the official Victorian suburb-group series, 2001–2025: 862 dwelling series, 26 candidates, scored the way a two-year forecast actually has to run — each guessed quarter fed back in as the next input.
>
> A few receipts:
>
> • Same model, three quizzes: $7.23 MAE on a shuffled 80/20 split (R² = 0.99). $12.04 on a one-step temporal split. $23.80 on the eight-quarter protocol. The first number is the one dashboards post. It is three times too optimistic for a two-year look-ahead.
> • A robust linear model (Huber on the quarterly change) won this grid at $23.80. Best XGBoost: $25.94. Last-quarter persistence — “assume rent does not change” — $25.98. The obvious baseline almost won.
> • Frozen 2025 audit: $14.87 vs $16.85 for persistence. About $2 a week. Stable on this panel. Small as a rent increment.
> • This is a group median, not the rent of an individual property.
>
> Predicting typical rent is useful in the near future. The 80% band tells you how far that usefulness stretches.
>
> Paper, code, and a fitted model you can query: https://github.com/kelvinvalani/vic-rent-ml
>
> Data: Homes Victoria Rental Report (RTBA bond lodgements), CC BY 4.0.

### Shorter variant

> Typical Brunswick 2-bed rent is $600 a week. Next quarter the 80% band is ±$17. Two years out it is ±$116. The further ahead you look, the worse a single dollar gets — the useful number is the range.
>
> That range is still worth having. Next quarter you can budget. A year out you can check a lease against the top of the band. Two years out you will not know the dollar, but you will know whether the suburb still works if typical rents run hot.
>
> Same model: $7 error on a shuffled split, $24 when you score eight quarters the way a real forecast has to run. Last-quarter persistence is within about $2 of the best learned model. The honest output is the band, not a fake-precise 633.
>
> Open write-up and model: https://github.com/kelvinvalani/vic-rent-ml
