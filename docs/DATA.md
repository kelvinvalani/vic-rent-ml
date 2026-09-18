# Data

## Source

The rental medians come from Homes Victoria's
[Rental Report — Quarterly: Moving Annual Rents by Suburb](https://discover.data.vic.gov.au/dataset/rental-report-quarterly-moving-annual-rents-by-suburb),
compiled from Residential Tenancies Bond Authority lodgements.

DataVic labels that dataset **Creative Commons Attribution 4.0 International**.
That licence is separate from the MIT licence on the code. If you redistribute
the CSVs, keep the Homes Victoria / RTBA attribution.

## Files in `data/`

| File | What it is |
|---|---|
| `VIC-data-transformed.csv` | Historical extraction used for the published experiment |
| `suburb-category.csv` | Postcode → region labels for optional regional grouping |
| `rent_panel.csv` | Modelling panel after lag joins (82,025 rows, 862 series, 2001 Q1–2025 Q3) |
| `sha256.json` | SHA-256 of those CSVs with newlines normalised to `\n` |

The published benchmark uses these files, not a freshly downloaded workbook.
A new official workbook can be ingested with `vic-rent-ml ingest --rent ...`;
that is a new experiment, not a bit-for-bit reproduction.

## Grain

- **Suburb group:** the named Homes Victoria area, often several suburbs.
- **Dwelling:** `flat` or `house`, bedrooms 1–4 (the published tables).
- **Target:** moving-annual median weekly rent, Australian dollars.
- **`postcodes`:** a representative postcode for serving. Some groups have none.

## What is not in the winning model

`lease_count` is present on the panel for documentation only. Amenity and macro
columns may also be present on `rent_panel.csv`; the bake-off does not use them.
See the [research paper](RESEARCH_PAPER.md) for why.
