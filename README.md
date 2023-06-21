# Generation of LIBS Data

This is a simple data generation script for optical emission spectra, issued from a [Laser-Induced Breakdown Spectroscopy](https://en.wikipedia.org/wiki/Laser-induced_breakdown_spectroscopy) (LIBS) experiment.

## Installation

Simply create an environment `python -m venv venv/` and activate it with `source venv/bin/activate`. Then use the `requirements.txt` to install the packages:

```bash
pip install -r requirements.txt
```

## Generation

The script `generate.py` handles the synthesis of LIBS spectra, provided some theoretical spectra, obtained using the [NIST database](https://physics.nist.gov/PhysRefData/ASD/LIBS/libs-form.html).

Use `python generate.py --help` to get help and the complete commands.

## Example

To generate 10000 LIBS spectra, with specific noise parameters:

```bash
python generate.py --input data/example_01.csv --num 10000 --beta 0.5 --gamme 0.3 --seed 123
```
