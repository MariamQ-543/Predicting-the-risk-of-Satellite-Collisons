#   SGP4 library: https://pypi.org/project/sgp4/
#   TLE format: https://en.wikipedia.org/wiki/Two-line_element_set
#   Julian dates: https://en.wikipedia.org/wiki/Julian_day

from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os

def parse_tle(tle_lines):
    """
    Parse TLE data into SGP4 satellite objects.
    
    TLE format is 3 lines per satellite:
    - Line 0: name
    - Line 1: epoch, drag, mean motion derivative
    - Line 2: inclination, eccentricity, argsument of perigee
    """
    satellites = []
    
    for i in range(0, len(tle_lines), 3):
        if i + 2 >= len(tle_lines):
            break
        
        name = tle_lines[i]
        line1 = tle_lines[i + 1]
        line2 = tle_lines[i + 2]
        
        sat = Satrec.twoline2rv(line1, line2)
        
        satellites.append({
            "name": name,
            "satellite": sat
        })
    
    return satellites


def propagate_satellite_at_time(satellite, target_time):
    """
    Propagate satellite to a specific datetime.
    
    Args:
        satellite: Satrec object
        target_time: datetime object for when to propagate to
    
    Returns:
        position: (x, y, z) in km (Earth-centered inertial frame)
        velocity: (vx, vy, vz) in km/s
    """
    # Convert to Julian date (the format SGP4 needs)
    jd, fr = jday(
        target_time.year,
        target_time.month,
        target_time.day,
        target_time.hour,
        target_time.minute,
        target_time.second
    )
    
    # Propagate using SGP4
    error_code, position, velocity = satellite.sgp4(jd, fr)
    
    if error_code != 0:
        return None, None
    
    return np.array(position), np.array(velocity)


def detect_conjunction(sat1, sat2, target_time):
    """
    Calculate the distance between two satellites at a specific time.
    
    Args:
        sat1, sat2: Satrec objects
        target_time: datetime object for when to check
    
    Returns:
        distance: separation distance in km (or None if propagation fails)
    """
    # Propagate both to exact same time
    pos1, _ = propagate_satellite_at_time(sat1, target_time)
    pos2, _ = propagate_satellite_at_time(sat2, target_time)
    
    if pos1 is None or pos2 is None:
        return None
    
    # Calculate distance between them
    distance = np.linalg.norm(pos1 - pos2)
    
    return distance


if __name__ == "__main__":
    from load_data import load_tle
    
    tle_lines = load_tle("space_track_tle.txt")
    sats = parse_tle(tle_lines)
    
    print(f"\nParsed {len(sats)} satellites")
    
    # Test propagation on first satellite
    print("\n" + "="*60)
    print("PROPAGATION TEST")
    print("="*60)
    print(f"\nSatellite: {sats[0]['name']}")
    
    # Set target time (24 hours from now)
    target_time = datetime.utcnow() + timedelta(hours=24)
    
    pos, vel = propagate_satellite_at_time(sats[0]['satellite'], target_time)
    
    if pos is not None:
        print(f"\nPosition at {target_time.strftime('%Y-%m-%d %H:%M')} UTC:")
        print(f"  x: {pos[0]:.2f} km")
        print(f"  y: {pos[1]:.2f} km")
        print(f"  z: {pos[2]:.2f} km")
        
        distance = np.linalg.norm(pos)
        speed = np.linalg.norm(vel)
        
        print(f"\nDistance from Earth center: {distance:.2f} km")
        print(f"Orbital speed: {speed:.2f} km/s")
        print("(LEO: ~6800-7200 km from center, ~7.5 km/s speed)")
    
    # Test conjunction detection on multiple pairs
    print("\n" + "="*60)
    print("CONJUNCTION DETECTION TEST (First 5 pairs)")
    print("="*60)
    
    results = []
    
    # Check first 5 satellite pairs
    for i in range(5):
        sat1_name = sats[i]['name']
        sat2_name = sats[i+1]['name']
        
        distance = detect_conjunction(
            sats[i]['satellite'],
            sats[i+1]['satellite'],
            target_time
        )
        
        if distance is not None:
            results.append({
                'sat1': sat1_name,
                'sat2': sat2_name,
                'target_time': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                'distance_km': distance
            })
            
            status = "⚠️ CLOSE" if distance < 10 else "✓ OK"
            print(f"{sat1_name} <-> {sat2_name}: {distance:.2f} km {status}")
    
    print(f"\nChecked {len(results)} satellite pairs")
    if results:
        print(f"Closest approach: {min(r['distance_km'] for r in results):.2f} km")
    
    # Save results to CSV
    os.makedirs("results/tables", exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/tables/sgp4_pair_distances.csv", index=False)
    print("\nSaved results to results/tables/sgp4_pair_distances.csv")