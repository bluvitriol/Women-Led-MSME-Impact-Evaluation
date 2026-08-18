# Women-Led MSME Impact Evaluation Pipeline

## Overview
A reproducible pipeline simulating an RCT impact evaluation of financial access interventions on women-led MSMEs.

## Data
Synthetic dataset generated to mirror real IE survey structure.

## How to Reproduce
1. Run code/01_simulate_data.py
2. Run code/02_clean_data.py
3. Run code/03_analysis.py
4. Run code/04_visualize.py
OR: Open notebooks/full_pipeline.ipynb

## Key Scripts
- [Data Cleaning](code/02_clean_data.py)
- [Analysis](code/03_analysis.py)

## Methods
OLS regression with treatment dummy. Balance checks via t-tests.
