import pandas as pd
import numpy as np
from load_data import load_csv

train = load_csv("esa_train.csv")

# Basic overview
print("\n" + "="*60)
print("BASIC OVERVIEW")
print("="*60)
print(f"Rows: {train.shape[0]}")
print(f"Columns: {train.shape[1]}")
print(f"\nUnique conjunction events: {train['event_id'].nunique()}")
print(f"Unique missions: {train['mission_id'].nunique()}")

# Target variable
print("\n" + "="*60)
print("TARGET VARIABLE: risk")
print("="*60)
print(train['risk'].describe())
print(f"\nMissing values in risk: {train['risk'].isna().sum()}")

# Missing values
print("\n" + "="*60)
print("MISSING VALUES (columns with any missing)")
print("="*60)
missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Data types
print("\n" + "="*60)
print("COLUMN DATA TYPES")
print("="*60)
print(train.dtypes.value_counts())

# Key feature distributions
print("\n" + "="*60)
print("KEY FEATURES - SUMMARY STATS")
print("="*60)
key_cols = ['miss_distance', 'relative_speed', 'time_to_tca',
            'mahalanobis_distance', 'risk']
print(train[key_cols].describe())

# Rows per event (time series structure)
print("\n" + "="*60)
print("ROWS PER EVENT (time series length)")
print("="*60)
rows_per_event = train.groupby('event_id').size()
print(rows_per_event.describe())
print(f"\nMin rows per event: {rows_per_event.min()}")
print(f"Max rows per event: {rows_per_event.max()}")