import os
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def load_csv(name, encoding="utf-8"):
    path = os.path.join(RAW_DIR, name)
    print(f"\nLoading {name}...")
    try:
        df = pd.read_csv(path, encoding=encoding)
    except UnicodeDecodeError:
        print(f"{name}: utf-8 failed, trying latin-1...")
        df = pd.read_csv(path, encoding="latin-1")
    print(f"\n{name} INFO:")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    return df

def load_tle(name):
    path = os.path.join(RAW_DIR, name)
    print(f"\nLoading {name}...")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"\n{name} INFO:")
    print("Total lines:", len(lines))
    print("Estimated satellites:", len(lines) // 3)
    print("\nFirst 6 lines:")
    for line in lines[:6]:
        print(line)
    return lines

if __name__ == "__main__":
    # ESA conjunction dataset - main data for training and testing models
    esa_train = load_csv("esa_train.csv")
    esa_test = load_csv("esa_test.csv")

    # Space weather data - solar and geomagnetic activity
    space_weather = load_csv("space_weather.csv")

    # Satellite physical properties - latin-1 encoding needed for special characters
    sat_props = load_csv("satellite_properties.csv", encoding="latin-1")

    # TLE data - raw orbital elements used for SGP4 physics baseline
    tle_lines = load_tle("space_track_tle.txt")