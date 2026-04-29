# Satellite Collision Risk Prediction Dashboard
# 3D visualisation of ESA mission orbits and conjunction risk
# orbit paths are drawn from representative orbital properties in the test data
# selecting a satellite shows its properties and most dangerous conjunction events
#
# run from src folder: streamlit run src/dashboard.py
#
# https://docs.streamlit.io/
# https://plotly.com/python/3d-charts/
# https://plotly.com/python/plotly-express/

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(
    page_title="Satellite Collision Risk",
    layout="wide",
    initial_sidebar_state="expanded"
)

# dark space theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
    .stApp { background-color: #020818; color: #e0e8f0; font-size: 16px; }
    .main-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem; color: #4fc3f7;
        letter-spacing: 0.05em; margin-bottom: 0;
    }
    .sub-title {
        font-family: 'Exo 2', sans-serif;
        font-size: 0.9rem; color: #78909c;
        margin-top: 0.2rem; margin-bottom: 1.5rem;
    }
    .section-header {
        font-family: 'Share Tech Mono', monospace;
        color: #4fc3f7; font-size: 1rem;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 0.3rem; margin-bottom: 1rem;
    }
    .legend-box {
        background: #0d1b2a; border: 1px solid #1e3a5f;
        border-radius: 6px; padding: 0.8rem; margin-bottom: 1rem;
    }
    .legend-item { display: flex; align-items: center; margin: 0.4rem 0; font-size: 0.85rem; }
    .dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
    .line { width: 24px; height: 3px; display: inline-block; margin-right: 8px; border-radius: 2px; }
    [data-testid="stSidebar"] {
        background-color: #030d1a;
        border-right: 1px solid #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    # load prediction files for all 5 models
    pred_paths = {
        'LightGBM':        'results/predictions/lightgbm_predictions.csv',
        'BiLSTM':          'results/predictions/lstm_predictions.csv',
        'MLP':             'results/predictions/mlp_predictions.csv',
        'GNN':             'results/predictions/gnn_predictions.csv',
        'PhysicsBaseline': 'results/predictions/physics_baseline_predictions.csv',
    }
    preds = {}
    for model, path in pred_paths.items():
        if os.path.exists(path):
            preds[model] = pd.read_csv(path)

    base = preds['LightGBM'][['event_id', 'actual_risk']].copy()
    for model, df in preds.items():
        base[f'pred_{model}'] = df['predicted_risk'].values

    # load ESA test data for orbital properties
    test_df = None
    for path in ['data/raw/esa_test.csv.csv', 'esa_test.csv', 'data/raw/esa_test.csv']:
        if os.path.exists(path):
            test_df = pd.read_csv(path)
            break

    metrics = pd.read_csv('results/tables/model_comparison_summary.csv')
    return base, preds, test_df, metrics


@st.cache_data
def build_event_table(test_df, pred_df):
    # get final CDM per event and merge with predictions
    if test_df is None:
        return pred_df

    final_cdm = (test_df.sort_values('time_to_tca')
                 .groupby('event_id').first().reset_index())

    cols = ['event_id', 'mission_id', 'miss_distance', 'relative_speed',
            'c_object_type', 't_j2k_sma', 't_j2k_ecc', 't_j2k_inc',
            'relative_position_r', 'relative_position_t', 'relative_position_n',
            'mahalanobis_distance', 'c_sigma_t',
            't_h_apo', 't_h_per', 't_rcs_estimate', 't_cd_area_over_mass',
            'c_j2k_sma', 'c_j2k_ecc', 'c_j2k_inc',
            'c_h_apo', 'c_h_per', 'c_rcs_estimate', 'c_cd_area_over_mass']
    cols = [c for c in cols if c in final_cdm.columns]
    return pred_df.merge(final_cdm[cols], on='event_id', how='left')


@st.cache_data
def compute_mission_stats(events_df):
    # compute per-mission risk statistics used to colour the orbits
    if 'mission_id' not in events_df.columns:
        return pd.DataFrame()

    stats = events_df.groupby('mission_id').agg(
        total_events=('actual_risk', 'count'),
        mean_risk=('actual_risk', 'mean'),
        min_risk=('actual_risk', 'min'),
        high_risk_events=('actual_risk', lambda x: (x > -8).sum()),
        sma=('t_j2k_sma', 'median'),
        ecc=('t_j2k_ecc', 'median'),
        inc=('t_j2k_inc', 'median')
    ).reset_index()
    stats['mission_id'] = stats['mission_id'].astype(int)
    return stats


def orbit_colour_by_risk(mean_risk):
    # colour the orbit line based on the satellite's average conjunction risk
    if mean_risk > -8:
        return '#ef5350'   # red - high average risk
    elif mean_risk > -12:
        return '#ffa726'   # orange - medium average risk
    else:
        return '#66bb6a'   # green - low average risk


def risk_label(risk_val):
    # consistent thresholds used everywhere
    if risk_val > -8:
        return 'HIGH'
    elif risk_val > -15:
        return 'MEDIUM'
    else:
        return 'LOW'


def make_orbit_points(sma_km, inc_deg, raan_deg=0, n_points=150):
    # draw a simple circular orbit around Earth in 3D
    # uses altitude (sma) from the dataset for orbit size
    # and inclination to tilt it at the right angle
    # orbits are circles not ellipses - accurate enough since ESA satellites have very low eccentricity
    #
    # https://plotly.com/python/3d-scatter-plots/
    # https://community.plotly.com/t/plot-earth-in-a-scatter3d-plot-for-orbits/44054
    # https://plainenglish.io/blog/plot-satellites-real-time-orbits-with-python-s-matplotlib

    inc  = np.radians(inc_deg)
    raan = np.radians(raan_deg)

    theta = np.linspace(0, 2 * np.pi, n_points)

    x_orb = sma_km * np.cos(theta)
    y_orb = sma_km * np.sin(theta)

    # tilt by inclination
    y_tilted = y_orb * np.cos(inc)
    z_tilted = y_orb * np.sin(inc)

    # rotate around z axis by RAAN to spread orbits around Earth
    x_fin = x_orb * np.cos(raan) - y_tilted * np.sin(raan)
    y_fin = x_orb * np.sin(raan) + y_tilted * np.cos(raan)
    z_fin = z_tilted

    return x_fin, y_fin, z_fin


def satellite_position(sma_km, inc_deg, raan_deg, phase):
    # get a single point on the circular orbit at a given phase angle
    # used to place the satellite marker and conjunction event dots
    inc  = np.radians(inc_deg)
    raan = np.radians(raan_deg)

    xo = sma_km * np.cos(phase)
    yo = sma_km * np.sin(phase)

    y_tilted = yo * np.cos(inc)
    z_tilted = yo * np.sin(inc)

    xf = xo * np.cos(raan) - y_tilted * np.sin(raan)
    yf = xo * np.sin(raan) + y_tilted * np.cos(raan)

    return xf, yf, z_tilted


def build_globe(mission_stats, events_df, selected_mission=None):
    fig = go.Figure()

    # Earth sphere with approximate land/sea colouring
    u = np.linspace(0, 2 * np.pi, 80)
    v = np.linspace(0, np.pi, 80)
    er = 6371
    xe = er * np.outer(np.cos(u), np.sin(v))
    ye = er * np.outer(np.sin(u), np.sin(v))
    ze = er * np.outer(np.ones(len(u)), np.cos(v))
    lat = np.outer(np.ones(len(u)), np.linspace(-90, 90, len(v)))
    sc  = np.sin(np.radians(lat)) * 0.5 + 0.5

    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        surfacecolor=sc,
        colorscale=[
            [0.0, '#051428'], [0.2, '#0d3060'],
            [0.4, '#1a5a9a'], [0.55, '#1e6b3a'],
            [0.7, '#2d8a4e'], [0.85, '#8b6914'], [1.0, '#d4c8a0']
        ],
        showscale=False, opacity=1.0,
        name='Earth', hoverinfo='skip'
    ))

    if mission_stats.empty:
        fig.update_layout(
            scene=dict(bgcolor='#020818',
                       xaxis=dict(showgrid=False, showticklabels=False, title=''),
                       yaxis=dict(showgrid=False, showticklabels=False, title=''),
                       zaxis=dict(showgrid=False, showticklabels=False, title=''),
                       aspectmode='cube'),
            paper_bgcolor='#020818',
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False, height=620
        )
        return fig

    missions  = sorted(mission_stats['mission_id'].tolist())
    raan_step = 360 / len(missions)
    np.random.seed(42)

    for i, row in mission_stats.iterrows():
        mission = int(row['mission_id'])
        sma     = float(row['sma']) if not pd.isna(row['sma']) else 7000.0
        ecc     = float(np.clip(row['ecc'] if not pd.isna(row['ecc']) else 0.001, 0, 0.95))
        inc     = float(row['inc']) if not pd.isna(row['inc']) else 98.0
        raan    = missions.index(mission) * raan_step
        colour  = orbit_colour_by_risk(row['mean_risk'])

        if sma < 6400:
            sma = 7000.0

        # small inclination offset per satellite to spread orbits visually
        # all ESA satellites have similar real inclinations so without this
        # they all look like the same polar orbit stacked on top of each other
        inc_offset  = (missions.index(mission) % 5) * 4
        display_inc = inc + inc_offset

        x, y, z = make_orbit_points(sma, display_inc, raan)

        is_selected = (selected_mission is None or mission == selected_mission)
        opacity     = 1.0 if is_selected else 0.15
        width       = 3.0 if is_selected else 0.8

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=colour, width=width),
            opacity=opacity,
            name=f'SAT-{mission:02d}',
            hovertemplate=(
                f'<b>SAT-{mission:02d}</b><br>'
                f'Mean Risk: {row["mean_risk"]:.2f} (log10)<br>'
                f'Total Events: {int(row["total_events"])}<br>'
                f'SMA: {sma:.0f} km<br>'
                f'Inclination: {inc:.1f} deg<extra></extra>'
            )
        ))

        phase = 0.4 * missions.index(mission)
        sx, sy, sz = satellite_position(sma, display_inc, raan, phase)

        fig.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode='markers+text',
            marker=dict(
                size=14 if is_selected else 4,
                color=colour,
                symbol='diamond',
                line=dict(color='white', width=2 if is_selected else 0)
            ),
            text=[f'SAT-{mission:02d}'],
            textposition='top center',
            textfont=dict(size=9 if is_selected else 7, color=colour),
            opacity=opacity,
            name=f'SAT-{mission:02d} pos',
            hovertemplate=(
                f'<b>SAT-{mission:02d}</b><br>'
                f'SMA: {sma:.0f} km<br>'
                f'Eccentricity: {ecc:.4f}<br>'
                f'Inclination: {inc:.1f} deg<extra></extra>'
            ),
            showlegend=False
        ))

        # show top 10 MOST dangerous events (descending = highest risk first)
        # higher risk value = more dangerous (less negative)
        if is_selected and selected_mission is not None and events_df is not None:
            sat_events = events_df[events_df['mission_id'].astype(int) == mission]
            sat_events = sat_events.sort_values('actual_risk', ascending=False).head(10)

            for _, ev in sat_events.iterrows():
                phase_ev = np.random.uniform(0, 2 * np.pi)
                ex, ey, ez = satellite_position(sma, display_inc, raan, phase_ev)

                risk     = ev['actual_risk']
                ec       = '#ef5350' if risk > -8 else '#ffa726' if risk > -15 else '#66bb6a'
                esize    = 12 if risk > -8 else 8 if risk > -15 else 5
                rl       = risk_label(risk)

                miss_d   = ev.get('miss_distance', float('nan'))
                speed    = ev.get('relative_speed', float('nan'))
                obj_type = ev.get('c_object_type', 'Unknown')
                lgbm     = ev.get('pred_LightGBM', float('nan'))

                miss_str  = f'{miss_d:.0f} m' if not pd.isna(miss_d) else 'N/A'
                speed_str = f'{speed:.0f} m/s' if not pd.isna(speed) else 'N/A'
                lgbm_str  = f'{lgbm:.2f}' if not pd.isna(lgbm) else 'N/A'

                fig.add_trace(go.Scatter3d(
                    x=[ex], y=[ey], z=[ez],
                    mode='markers',
                    marker=dict(size=esize, color=ec, opacity=0.95,
                                line=dict(color='white', width=0.5)),
                    name=f'Event {int(ev["event_id"])}',
                    hovertemplate=(
                        f'<b>Potential Collision Event</b><br>'
                        f'Risk Level: {rl}<br>'
                        f'Actual Risk: {risk:.2f} (log10)<br>'
                        f'LightGBM Prediction: {lgbm_str}<br>'
                        f'Miss Distance: {miss_str}<br>'
                        f'Relative Speed: {speed_str}<br>'
                        f'Object Type: {obj_type}<extra></extra>'
                    ),
                    showlegend=False
                ))

    fig.update_layout(
        scene=dict(
            bgcolor='#020818',
            xaxis=dict(showgrid=False, showticklabels=False, title='', backgroundcolor='#020818'),
            yaxis=dict(showgrid=False, showticklabels=False, title='', backgroundcolor='#020818'),
            zaxis=dict(showgrid=False, showticklabels=False, title='', backgroundcolor='#020818'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.2, z=0.7))
        ),
        paper_bgcolor='#020818',
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        height=620,
        hoverdistance=300,
        hovermode='closest'
    )
    return fig


# sidebar navigation
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["Orbit Visualisation", "Model Comparison",
         "Evaluation", "Explainability", "Mission Overview", "About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Dataset:** ESA Collision Avoidance Challenge")
    st.markdown("**Events:** 2,167 test conjunctions")
    st.markdown("**Satellites:** 19 ESA missions")
    st.markdown("**Models:** 5 compared")


# load data on startup
try:
    pred_df, preds, test_df, metrics_df = load_all_data()
    events_df     = build_event_table(test_df, pred_df)
    mission_stats = compute_mission_stats(events_df)
    data_loaded   = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    data_loaded = False


# page: orbit visualisation
if page == "Orbit Visualisation" and data_loaded:
    st.markdown('<p class="main-title">Satellite Collision Risk Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">ESA Collision Avoidance Challenge - Real operational conjunction data 2015-2019</p>', unsafe_allow_html=True)

    col_globe, col_panel = st.columns([2.2, 1])

    with col_panel:
        st.markdown('<p class="section-header">Map Key</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="legend-box">
            <div style="font-size:0.8rem; color:#78909c; margin-bottom:0.6rem;">Orbit colour = average collision risk for that satellite</div>
            <div class="legend-item"><span class="line" style="background:#ef5350;"></span> High average risk (> -8)</div>
            <div class="legend-item"><span class="line" style="background:#ffa726;"></span> Medium average risk</div>
            <div class="legend-item"><span class="line" style="background:#66bb6a;"></span> Low average risk</div>
            <div style="margin-top:0.8rem; font-size:0.8rem; color:#78909c;">Conjunction event dots (visible when satellite selected)</div>
            <div class="legend-item"><span class="dot" style="background:#ef5350;"></span> High risk event (risk > -8)</div>
            <div class="legend-item"><span class="dot" style="background:#ffa726;"></span> Medium risk event (-15 to -8)</div>
            <div class="legend-item"><span class="dot" style="background:#66bb6a;"></span> Low risk event (< -15)</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("What the properties mean"):
            st.markdown("""
            - **SMA (Altitude):** Average distance from Earth's centre in km. ~330-830 km above surface.
            - **Apogee / Perigee:** Highest and lowest points of the orbit above Earth's surface.
            - **Eccentricity:** How circular the orbit is. 0 = perfect circle.
            - **Inclination:** Orbit tilt relative to equator. ~98 deg = polar orbit.
            - **Miss Distance:** Closest predicted distance between the two objects in metres.
            - **Mahalanobis Distance:** Danger score accounting for orbit uncertainty. Smaller = more dangerous.
            - **Relative Speed:** How fast the two objects approach each other in m/s.
            - **Chaser Type:** DEBRIS = space junk, PAYLOAD = another satellite, ROCKET BODY = spent rocket stage.
            - **Actual Risk:** Log10 collision probability. Higher values are more dangerous. -4 = 1 in 10,000. ESA acts above -4.
            - **LightGBM Pred:** The best ML model's predicted final risk using early warning messages.
            """)

        st.markdown('<p class="section-header">Select Satellite</p>', unsafe_allow_html=True)
        missions    = sorted(mission_stats['mission_id'].tolist()) if not mission_stats.empty else []
        options     = ['All Satellites'] + [f'SAT-{m:02d}' for m in missions]
        selected    = st.selectbox("", options, label_visibility="collapsed")
        sel_mission = None if selected == 'All Satellites' else int(selected.split('-')[1])

    with col_globe:
        st.markdown('<p class="section-header">3D Earth and Satellite Orbits</p>', unsafe_allow_html=True)
        st.caption("Select a satellite from the panel to view its conjunction events on the orbit. Hover over the coloured dots for event details. Drag to rotate.")
        st.caption("Orbit lines are simplified visual representations based on representative orbital parameters from the ESA test data. Included for interpretation, not exact trajectory reconstruction.")
        fig = build_globe(mission_stats, events_df, sel_mission)
        st.plotly_chart(fig, use_container_width=True)

    # consistent risk counts using same thresholds as risk_label
    n_high = (events_df['actual_risk'] > -8).sum()
    n_med  = ((events_df['actual_risk'] <= -8) & (events_df['actual_risk'] > -15)).sum()
    n_low  = (events_df['actual_risk'] <= -15).sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Events", len(events_df))
    c2.metric("High Risk", int(n_high))
    c3.metric("Medium Risk", int(n_med))
    c4.metric("Low Risk", int(n_low))

    if sel_mission is not None:
        sat_row    = mission_stats[mission_stats['mission_id'] == sel_mission]
        sat_events = events_df[events_df['mission_id'].astype(int) == sel_mission].copy() \
                     if 'mission_id' in events_df.columns else pd.DataFrame()

        if not sat_row.empty:
            r = sat_row.iloc[0]
            st.markdown(f'<p class="section-header">SAT-{sel_mission:02d} — Satellite Properties</p>',
                        unsafe_allow_html=True)

            col_t, col_c = st.columns(2)
            with col_t:
                st.markdown("**Target Satellite (ESA Mission)**")
                tc1, tc2, tc3 = st.columns(3)
                tc1.metric("SMA", f"{r['sma']:.0f} km" if not pd.isna(r['sma']) else 'N/A')
                tc2.metric("Inclination", f"{r['inc']:.1f} deg" if not pd.isna(r['inc']) else 'N/A')
                tc3.metric("Eccentricity", f"{r['ecc']:.4f}" if not pd.isna(r['ecc']) else 'N/A')

                if len(sat_events) > 0:
                    tc4, tc5, tc6 = st.columns(3)
                    for col_name, label, fmt in [
                        ('t_h_apo', 'Apogee', '{:.0f} km'),
                        ('t_h_per', 'Perigee', '{:.0f} km'),
                        ('t_rcs_estimate', 'Radar Cross Section', '{:.2f} m2')
                    ]:
                        val = sat_events[col_name].median() if col_name in sat_events.columns else float('nan')
                        [tc4, tc5, tc6][[col_name == 't_h_apo', col_name == 't_h_per',
                                          col_name == 't_rcs_estimate'].index(True)].metric(
                            label, fmt.format(val) if not pd.isna(val) else 'N/A')

                    tc7, tc8 = st.columns(2)
                    aom = sat_events['t_cd_area_over_mass'].median() \
                          if 't_cd_area_over_mass' in sat_events.columns else float('nan')
                    tc7.metric("Area/Mass ratio", f"{aom:.4f} m2/kg" if not pd.isna(aom) else 'N/A')
                    tc8.metric("Total Events", int(r['total_events']))

            with col_c:
                st.markdown("**Summary**")
                st.metric("High Risk Events", int(r['high_risk_events']))
                st.metric("Mean Risk", f"{r['mean_risk']:.2f}")
                if len(sat_events) > 0 and 'miss_distance' in sat_events.columns:
                    st.metric("Min Miss Distance", f"{sat_events['miss_distance'].min():.0f} m")
                if len(sat_events) > 0 and 'relative_speed' in sat_events.columns:
                    st.metric("Mean Relative Speed", f"{sat_events['relative_speed'].mean():.0f} m/s")

        st.markdown(f'<p class="section-header">SAT-{sel_mission:02d} — All Conjunction Events</p>',
                    unsafe_allow_html=True)

        if len(sat_events) > 0:
            cdm_counts = pd.DataFrame()
            if test_df is not None and 'event_id' in test_df.columns:
                cdm_counts = test_df.groupby('event_id').agg(
                    n_cdms=('time_to_tca', 'count'),
                    first_warning_days=('time_to_tca', 'max')
                ).reset_index()

            if len(cdm_counts) > 0:
                sat_events = sat_events.merge(cdm_counts, on='event_id', how='left')

            col_map = {
                'event_id':             'Event ID',
                'actual_risk':          'Actual Risk',
                'pred_LightGBM':        'LightGBM Pred',
                'miss_distance':        'Miss Dist (m)',
                'mahalanobis_distance': 'Mahalanobis Dist',
                'relative_speed':       'Rel Speed (m/s)',
                'relative_position_r':  'Position R (m)',
                'relative_position_t':  'Position T (m)',
                'relative_position_n':  'Position N (m)',
                'c_object_type':        'Chaser Type',
                'c_j2k_sma':            'Chaser SMA (km)',
                'c_h_apo':              'Chaser Apogee (km)',
                'c_h_per':              'Chaser Perigee (km)',
                'c_rcs_estimate':       'Chaser RCS (m2)',
                'c_cd_area_over_mass':  'Chaser Area/Mass',
                'first_warning_days':   'First Warning (days)',
                'n_cdms':               'Num CDMs',
            }

            available = {k: v for k, v in col_map.items() if k in sat_events.columns}
            disp = sat_events[list(available.keys())].copy()
            disp = disp.rename(columns=available)
            disp['Risk Level'] = sat_events['actual_risk'].apply(risk_label)
            # sort descending - higher value = more dangerous
            disp = disp.sort_values('Actual Risk', ascending=False)
            st.dataframe(disp.round(2), use_container_width=True, height=500)
        else:
            st.info("No conjunction data available for this satellite.")

    else:
        st.markdown('<p class="section-header">Top 10 Most Dangerous Conjunction Events (All Satellites)</p>',
                    unsafe_allow_html=True)
        st.caption("Higher risk values are more dangerous because the values are log10 probabilities. For example, risk = -4 means 1 in 10,000, while risk = -30 is extremely low risk. Select a satellite from the dropdown above to see all conjunction events for that mission specifically.")

        # sort descending - highest (least negative) = most dangerous
        top = events_df.sort_values('actual_risk', ascending=False).head(10)
        col_map = {
            'event_id':             'Event ID',
            'mission_id':           'Satellite',
            'actual_risk':          'Actual Risk',
            'pred_LightGBM':        'LightGBM Pred',
            'miss_distance':        'Miss Dist (m)',
            'mahalanobis_distance': 'Mahalanobis Dist',
            'relative_speed':       'Rel Speed (m/s)',
            'c_object_type':        'Chaser Type',
            'c_j2k_sma':            'Chaser SMA (km)',
        }
        available = {k: v for k, v in col_map.items() if k in top.columns}
        disp = top[list(available.keys())].copy()
        disp = disp.rename(columns=available)
        if 'Satellite' in disp.columns:
            disp['Satellite'] = disp['Satellite'].apply(
                lambda x: f'SAT-{int(x):02d}' if not pd.isna(x) else 'N/A')
        st.dataframe(disp.round(2), use_container_width=True, height=400)

    if not mission_stats.empty:
        st.markdown('<p class="section-header">All Satellites Ranked by Average Collision Risk</p>',
                    unsafe_allow_html=True)
        rank_df = mission_stats.sort_values('mean_risk', ascending=False).copy()
        rank_df['satellite'] = rank_df['mission_id'].apply(lambda x: f'SAT-{int(x):02d}')
        rank_df['colour']    = rank_df['mean_risk'].apply(orbit_colour_by_risk)

        fig_rank = go.Figure(go.Bar(
            x=rank_df['satellite'],
            y=rank_df['mean_risk'],
            marker_color=rank_df['colour'].tolist(),
            hovertemplate='<b>%{x}</b><br>Mean Risk: %{y:.2f}<extra></extra>'
        ))
        fig_rank.update_layout(
            paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
            font_color='#e0e8f0',
            yaxis_title='Mean Risk (log10)',
            xaxis_title='Satellite',
            height=260,
            margin=dict(l=40, r=20, t=10, b=40)
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        st.caption("Higher value = more dangerous. Red = high risk, Orange = medium, Green = low.")


# page: model comparison
elif page == "Model Comparison" and data_loaded:
    st.markdown('<p class="main-title">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Performance of all 5 models on 2,167 test conjunction events</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Performance Metrics</p>', unsafe_allow_html=True)
    st.dataframe(metrics_df.round(4), use_container_width=True)
    st.caption("R² shows how well each model predicts risk. 1 is perfect, negative means worse than guessing. LightGBM wins across every metric by a clear margin.")

    col1, col2 = st.columns(2)
    with col1:
        fig_r2 = px.bar(
            metrics_df.sort_values('R2'),
            x='R2', y='Model', orientation='h',
            color='R2',
            color_continuous_scale=['#ef5350', '#ffa726', '#66bb6a'],
            title='R2 Score (higher is better)'
        )
        fig_r2.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
                              font_color='#e0e8f0', coloraxis_showscale=False)
        st.plotly_chart(fig_r2, use_container_width=True)
        st.caption("Higher R² means the model explains more of the variation in risk. 1.0 is perfect, 0 means no better than guessing the average.")

    with col2:
        fig_rmse = px.bar(
            metrics_df.sort_values('RMSE', ascending=False),
            x='RMSE', y='Model', orientation='h',
            color='RMSE',
            color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
            title='RMSE (lower is better)'
        )
        fig_rmse.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
                               font_color='#e0e8f0', coloraxis_showscale=False)
        st.plotly_chart(fig_rmse, use_container_width=True)
        st.caption("Lower RMSE is better. In log10 risk units, a difference of 1 means the prediction is off by one order of magnitude.")

    st.markdown('<p class="section-header">Predicted vs Actual Risk</p>', unsafe_allow_html=True)
    model_select = st.selectbox("Select model", list(preds.keys()))
    df_plot = preds[model_select]
    fig_sc = px.scatter(
        df_plot, x='actual_risk', y='predicted_risk', opacity=0.4,
        title=f'{model_select} - Predicted vs Actual Risk',
        labels={'actual_risk': 'Actual Risk (log10)', 'predicted_risk': 'Predicted Risk (log10)'}
    )
    lim = [df_plot[['actual_risk', 'predicted_risk']].min().min() - 1,
           df_plot[['actual_risk', 'predicted_risk']].max().max() + 1]
    fig_sc.add_shape(type='line', x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1],
                     line=dict(color='white', dash='dash', width=1))
    fig_sc.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a', font_color='#e0e8f0')
    st.plotly_chart(fig_sc, use_container_width=True)
    st.caption("Points close to the dashed line mean accurate predictions. LightGBM clusters tightly around it. The GNN and physics baseline show wide scatter, confirming poor prediction accuracy.")


# page: evaluation
elif page == "Evaluation" and data_loaded:
    st.markdown('<p class="main-title">Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Detailed evaluation plots for all 5 models</p>', unsafe_allow_html=True)

    # risk distribution
    st.markdown('<p class="section-header">Risk Distribution in Test Set</p>', unsafe_allow_html=True)
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=pred_df['actual_risk'],
        nbinsx=50,
        marker_color='#4fc3f7',
        opacity=0.8,
        name='Actual Risk'
    ))
    fig_dist.update_layout(
        paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
        font_color='#e0e8f0',
        xaxis_title='Risk (log10)',
        yaxis_title='Number of Events',
        height=300,
        margin=dict(l=40, r=20, t=10, b=40)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption("Distribution of actual risk values across all 2,167 test events. Most events cluster at very low risk values, which shows why the problem is hard. High risk events are rare but are the ones that matter most operationally.")

    # error distribution per model
    st.markdown('<p class="section-header">Prediction Error Distribution by Model</p>', unsafe_allow_html=True)
    fig_err = go.Figure()
    colours = {
        'LightGBM': '#4fc3f7',
        'MLP': '#ffa726',
        'BiLSTM': '#66bb6a',
        'GNN': '#ef5350',
        'PhysicsBaseline': '#ab47bc'
    }
    for model, df in preds.items():
        errors = df['predicted_risk'] - df['actual_risk']
        fig_err.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name=model,
            marker_color=colours.get(model, '#ffffff'),
            opacity=0.6
        ))
    fig_err.update_layout(
        barmode='overlay',
        paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
        font_color='#e0e8f0',
        xaxis_title='Prediction Error (predicted - actual)',
        yaxis_title='Number of Events',
        height=350,
        margin=dict(l=40, r=20, t=10, b=40),
        legend=dict(bgcolor='#0d1b2a', bordercolor='#1e3a5f')
    )
    st.plotly_chart(fig_err, use_container_width=True)
    st.caption("Prediction error is predicted risk minus actual risk. Values close to 0 mean accurate predictions. LightGBM is closest to 0 for most events, while the GNN and physics baseline have much wider errors.")

    # performance heatmap from saved plot
    st.markdown('<p class="section-header">Performance Heatmap</p>', unsafe_allow_html=True)
    if os.path.exists('results/plots/model_heatmap.png'):
        st.image('results/plots/model_heatmap.png',
                 caption='Combined view of R², RMSE and MAE across all models. Green = better, red = worse. LightGBM is best across every metric.')
    else:
        st.info("Heatmap not found. Run evaluate_models.py to generate.")

    # predicted vs actual all models
    st.markdown('<p class="section-header">Predicted vs Actual — All Models</p>', unsafe_allow_html=True)
    if os.path.exists('results/plots/predicted_vs_actual_all.png'):
        st.image('results/plots/predicted_vs_actual_all.png',
                 caption='Each subplot shows predicted vs actual risk for one model. Points close to the dashed line mean accurate predictions. LightGBM follows the diagonal most closely.')
    else:
        st.info("Plot not found. Run evaluate_models.py to generate.")

    # residuals all models
    st.markdown('<p class="section-header">Residual Plots — All Models</p>', unsafe_allow_html=True)
    if os.path.exists('results/plots/residuals_all.png'):
        st.image('results/plots/residuals_all.png',
                 caption='Residuals are prediction errors. A good model should have errors scattered around 0 with no clear pattern. Clear patterns mean the model is getting some cases wrong in a systematic way.')
    else:
        st.info("Plot not found. Run evaluate_models.py to generate.")

    # r2 and rmse comparison plots
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('results/plots/model_comparison_r2.png'):
            st.image('results/plots/model_comparison_r2.png',
                     caption='R² comparison across all models.')
    with col2:
        if os.path.exists('results/plots/model_comparison_rmse.png'):
            st.image('results/plots/model_comparison_rmse.png',
                     caption='RMSE comparison across all models.')


# page: explainability
elif page == "Explainability" and data_loaded:
    st.markdown('<p class="main-title">Model Explainability</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">SHAP analysis for LightGBM - understanding what drives each prediction</p>', unsafe_allow_html=True)

    st.markdown("""
    SHAP shows how much each feature contributed to each prediction.
    A positive SHAP value pushes the predicted risk higher. A negative value pushes it lower.
    """)

    shap_path = 'results/tables/shap_feature_importance.csv'
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path).head(20)
        fig_shap = px.bar(
            shap_df.sort_values('mean_abs_shap'),
            x='mean_abs_shap', y='feature', orientation='h',
            color='mean_abs_shap',
            color_continuous_scale=['#1e3a5f', '#4fc3f7'],
            title='LightGBM - Top 20 Features by Mean |SHAP Value|',
            labels={'mean_abs_shap': 'Mean |SHAP Value|', 'feature': 'Feature'}
        )
        fig_shap.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
                               font_color='#e0e8f0', coloraxis_showscale=False, height=600)
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption("The top features are all about orbit uncertainty and geometry. Space weather barely matters — atmospheric conditions have very little influence on the final collision probability.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Top predictors:**
            - c_sigma_t: chaser along-track position uncertainty
            - mahalanobis_distance: statistical separation accounting for orbit uncertainty
            - relative_position_r: radial separation at closest approach
            - miss_distance: minimum distance between the two objects
            """)
        with col2:
            st.markdown("""
            **Key finding:**
            Space weather features (F10, F3M, AP) rank near the bottom.
            Orbital uncertainty and conjunction geometry matter far more
            than atmospheric conditions when predicting final collision risk.
            """)

    st.markdown('<p class="section-header">SHAP Plots</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists('results/plots/shap_summary_bar.png'):
            st.image('results/plots/shap_summary_bar.png',
                     caption='Overall feature importance. Longer bar means that feature had more impact on predictions.')
    with c2:
        if os.path.exists('results/plots/shap_beeswarm.png'):
            st.image('results/plots/shap_beeswarm.png',
                     caption='SHAP Beeswarm Plot — each dot is one event. Red dots pushed to the right mean high feature values increase predicted risk.')
    if os.path.exists('results/plots/shap_waterfall_high_risk.png'):
        st.image('results/plots/shap_waterfall_high_risk.png',
                 caption='Prediction breakdown for the highest risk event in the test set. Each bar shows how much one feature pushed the prediction up or down.')


# page: mission overview
elif page == "Mission Overview" and data_loaded:
    st.markdown('<p class="main-title">Mission Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Collision risk statistics across 19 ESA missions</p>', unsafe_allow_html=True)

    if not mission_stats.empty:
        ms = mission_stats.copy()
        ms['satellite'] = ms['mission_id'].apply(lambda x: f'SAT-{int(x):02d}')
        ms = ms.sort_values('mean_risk', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig_mean = px.bar(ms, x='satellite', y='mean_risk',
                              color='mean_risk',
                              color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                              title='Average Risk per Satellite (higher = more dangerous)',
                              labels={'satellite': 'Satellite', 'mean_risk': 'Mean Risk (log10)'})
            fig_mean.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
                                   font_color='#e0e8f0', coloraxis_showscale=False)
            st.plotly_chart(fig_mean, use_container_width=True)
            st.caption("Higher value = more dangerous. Red satellites face the most dangerous conjunctions on average.")

        with col2:
            fig_cnt = px.bar(ms.sort_values('total_events', ascending=False),
                             x='satellite', y='total_events',
                             color='high_risk_events',
                             color_continuous_scale=['#1e3a5f', '#ef5350'],
                             title='Conjunction Events per Satellite',
                             labels={'satellite': 'Satellite', 'total_events': 'Total Events',
                                     'high_risk_events': 'High Risk Events'})
            fig_cnt.update_layout(paper_bgcolor='#020818', plot_bgcolor='#0d1b2a',
                                  font_color='#e0e8f0', coloraxis_showscale=False)
            st.plotly_chart(fig_cnt, use_container_width=True)
            st.caption("Total conjunction events per satellite.")

        st.markdown('<p class="section-header">Full Statistics Table</p>', unsafe_allow_html=True)
        st.dataframe(ms.drop('mission_id', axis=1).round(3), use_container_width=True)


# page: about
elif page == "About" and data_loaded:
    st.markdown('<p class="main-title">About This Project</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">What This Project Does</p>', unsafe_allow_html=True)
    st.markdown("""
    Predicting how likely a satellite collision is before it happens. ESA gets warning
    messages called CDMs when two space objects are on a close approach path. These
    arrive up to 7 days before closest approach, each one updating the estimated
    collision probability as tracking improves. The problem is the final probability
    is only known at the very last message, too late to react comfortably. The models
    predict that final probability early using the messages available so far, giving
    operators more time to decide whether to move the satellite.
    """)

    st.markdown('<p class="section-header">The Data</p>', unsafe_allow_html=True)
    st.markdown("""
    The ESA Collision Avoidance Challenge dataset, real warning messages from ESA's
    Space Debris Office covering 2015-2019. 162,634 messages across 13,154 close approach
    events involving 19 ESA satellites. Each event has roughly 12 messages sent over 7 days.

    The ESA dataset contains a risk column, which is the log10 collision probability
    calculated by ESA at the time of each warning message. Log10 is just a compact way
    of writing very small numbers. Higher values are more dangerous — risk = -4 means
    1 in 10,000, risk = -6 means 1 in 1,000,000. ESA considers a manoeuvre when risk
    exceeds -4. Our models predict what this value will be at the final message, using
    only the earlier messages available so far.

    Dataset: https://kelvins.esa.int/collision-avoidance-challenge/data/
    """)

    st.markdown('<p class="section-header">Models Built and Why</p>', unsafe_allow_html=True)
    st.markdown("""
    All 5 models evaluated fairly on the same 2,167 test events.

    **Physics Baseline: R2=-2.34**
    A simple baseline using ESA's max_risk_estimate field. This represents an
    estimate of collision probability, not the final risk. It performed poorly, showing that
    early estimates alone are not enough to predict final risk.
    How: used the max_risk_estimate column directly as the prediction with no ML involved.

    **LightGBM: R2=0.856**
    Chosen because gradient boosting methods performed strongly in the ESA competition.
    How: trained on 98 features from the final CDM per event using 200 trees and a
    learning rate of 0.05. Removed two leaky columns that gave a fake R2=0.9998.

    **MLP: R2=0.488**
    Added to separate two things, whether sequential structure in BiLSTM adds value over
    a flat neural network, and whether neural networks in general can compete with tree
    models on this data. Same features as LightGBM but learns through neural layers
    instead of decision trees.
    How: built a 4 hidden layer network (256, 128, 64, 32) with BatchNorm and Dropout
    to stabilise training and prevent overfitting. Trained with Adam optimiser.

    **BiLSTM with Attention: R2=0.237**
    A normal LSTM reads CDMs in one direction only, so it only knows what came before
    each message. BiLSTM reads both forwards and backwards, giving each timestep access
    to information from the whole sequence. The attention mechanism then learns which
    CDMs matter most.
    How: each event becomes a padded sequence of up to 23 CDMs. Two BiLSTM layers
    extract temporal patterns, an attention layer scores each timestep, and the
    weighted sum feeds into a final prediction layer.

    **GNN: R2=-0.356**
    Added to explore how graph-based learning compares to tree and sequence models on this data.
    Each conjunction event is a node, with edges connecting events from the same satellite
    sorted by time. Two GCN layers do message passing so each event learns from nearby
    events before predicting risk. Performance was poor because conjunction events are largely
    independent, so averaging neighbouring events often introduced noise rather than useful
    additional information. The standalone MLP performed much better using the same features
    without graph structure, confirming the data does not naturally suit graph-based learning.
    How: built a graph with 13,154 training nodes. Each node has 98 CDM features.
    Edges connect events from the same mission sorted by time_to_tca, each event
    connected to its 5 nearest neighbours. Two GCNConv layers then a small MLP head.
    """)

    st.markdown('<p class="section-header">Key Findings</p>', unsafe_allow_html=True)
    st.markdown("""
    LightGBM did a lot better than all the other ML models. A tree-based model also won
    the original ESA competition in 2019 so this result was expected.

    Physics baseline R2=-2.34 vs LightGBM R2=0.856 shows why using ML benefits
    predicting the risk of collisions.

    SHAP analysis found that chaser position uncertainty is more predictive than raw
    miss distance. How precisely we know where the debris is matters more than where
    we think it is. Space weather features ranked near the bottom, showing orbital
    geometry matters far more than atmospheric conditions.

    Overall, the results show that increasing model complexity does not necessarily improve
    performance, and that this dataset is best suited to structured tabular modelling approaches.
    """)

    st.markdown('<p class="section-header">Results</p>', unsafe_allow_html=True)
    st.markdown("""
    | Model Type | Model | R2 | RMSE | MAE |
    |---|---|---|---|---|
    | Baseline | Physics Baseline | -2.34 | 18.29 | 15.50 |
    | Tabular ML | LightGBM | 0.856 | 3.80 | 2.52 |
    | Neural | MLP | 0.488 | 7.16 | 4.99 |
    | Sequence | BiLSTM | 0.237 | 6.77 | 4.72 |
    | Graph | GNN | -0.356 | 11.65 | 9.77 |
    """)

    st.markdown('<p class="section-header">Dataset</p>', unsafe_allow_html=True)
    st.markdown("""
    ESA Collision Avoidance Challenge Dataset: https://kelvins.esa.int/collision-avoidance-challenge/data/
    """)