from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()

# ── Paths ─────────────────────────────────────────────────────
DATA_PATH   = Path("Crime_Data_from_2020_to_Present.csv")
OUTPUT_HTML = Path("SYSTEM_C.html")
file_path   = str(DATA_PATH)   # shared with preprocessing block below

# ── Crime group order (System 1 canonical category names) ─────
CRIME_GROUP_ORDER = [
    "Violent Crime", "Property Crime", "Vehicle Crime",
    "Domestic Violence", "Fraud / Financial Crime",
    "Sexual Offenses", "Threats / Harassment",
    "Vandalism / Trespass", "Other",
]
AGE_BANDS = ["<18", "18-25", "26-40", "41-60", "60+"]

#imports
import pandas as pd
import numpy as np
import json

# =========================================================
# DATA LOADING & CLEANING  (System 1 preprocessing — identical steps)
# =========================================================

# Step 1: Load data
df_crime = pd.read_csv(file_path)

# Step 2: Drop irrelevant columns
df_crime.drop(['Cross Street', 'Mocodes'], axis=1, inplace=True)

# Step 3: Fill null values in secondary crime codes
df_crime['Crm Cd 1'] = df_crime['Crm Cd 1'].fillna('No Other Crime').astype(str)
df_crime['Crm Cd 2'] = df_crime['Crm Cd 2'].fillna('No Other Crime').astype(str)
df_crime['Crm Cd 3'] = df_crime['Crm Cd 3'].fillna('No Other Crime').astype(str)
df_crime['Crm Cd 4'] = df_crime['Crm Cd 4'].fillna('No Other Crime').astype(str)

# Step 4: Fill null values in weapon columns
df_crime['Weapon Used Cd'] = df_crime['Weapon Used Cd'].fillna(0)
df_crime['Weapon Desc']    = df_crime['Weapon Desc'].fillna('No Weapon')

# Step 5: Keep only valid ages (0–119)
df_crime = df_crime[(df_crime['Vict Age'] >= 0) & (df_crime['Vict Age'] < 120)]

# Step 6: Age-0 filter — keep age-0 rows only for child-related crimes
keep_crimes = [
    'CHILD NEGLECT (SEE 300 W.I.C.)',
    'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT',
    'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
    'KIDNAPPING',
    'KIDNAPPING - GRAND ATTEMPT',
    'CHILD STEALING',
    'CHILD ABANDONMENT',
]
df_crime = df_crime[
    (df_crime['Vict Age'] != 0) |
    ((df_crime['Vict Age'] == 0) & (df_crime['Crm Cd Desc'].isin(keep_crimes)))
]

# Step 7: Clean Vict Sex — replace invalid codes with Unknown
df_crime['Vict Sex'] = df_crime['Vict Sex'].replace({'X': np.nan, 'H': np.nan, '-': np.nan})
df_crime['Vict Sex'] = df_crime['Vict Sex'].fillna('Unknown')

# Step 8: Drop remaining NaN rows
df_crime.dropna(inplace=True)

# Step 9: Sample to 10000 rows
if len(df_crime) > 10000:
    df_crime = df_crime.sample(n=10000, random_state=42).copy()

# Step 10: Reset index
df_crime = df_crime.reset_index(drop=True)

# Step 11: Parse date column
df_crime['Date Rptd'] = pd.to_datetime(df_crime['Date Rptd'], errors='coerce')

# Step 12: Drop rows missing key fields
df_crime = df_crime.dropna(subset=['Date Rptd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex']).copy()

# Step 13: Crime category classifier (System 1 logic — identical keywords & labels)
def classify_crime(crime_description):
    crime_description = str(crime_description).lower()
    if 'rape' in crime_description or 'sexual' in crime_description:
        return 'Sexual Offenses'
    if 'identity' in crime_description or 'bunco' in crime_description:
        return 'Fraud / Financial Crime'
    if 'intimate partner' in crime_description or 'restraining order' in crime_description:
        return 'Domestic Violence'
    if 'threat' in crime_description or 'lewd' in crime_description:
        return 'Threats / Harassment'
    if 'vehicle' in crime_description:
        return 'Vehicle Crime'
    if 'vandalism' in crime_description or 'trespass' in crime_description:
        return 'Vandalism / Trespass'
    if any(w in crime_description for w in ['assault', 'battery', 'robbery', 'brandish']):
        return 'Violent Crime'
    if any(w in crime_description for w in ['burglary', 'theft', 'shoplifting', 'pickpocket', 'bike']):
        return 'Property Crime'
    return 'Other'

df_crime['crime_category'] = df_crime['Crm Cd Desc'].apply(classify_crime)

# Step 14: Derive date columns
df_crime['year_month'] = df_crime['Date Rptd'].dt.to_period('M').dt.to_timestamp()
df_crime['year']       = df_crime['Date Rptd'].dt.year.astype(str)
df_crime['Year']       = df_crime['Date Rptd'].dt.year.astype(int)
df_crime['MonthStart'] = df_crime['year_month']

# Step 15: Remove incomplete 2025 data
df_crime = df_crime[df_crime['year_month'].dt.year < 2025].copy()

# Step 16: Final age filter — keep only 0 < age < 100
df_crime = df_crime[(df_crime['Vict Age'] > 0) & (df_crime['Vict Age'] < 100)].copy()

# Step 17: Standardise Vict Sex labels
df_crime['Vict Sex'] = df_crime['Vict Sex'].replace({'M': 'Male', 'F': 'Female', 'X': 'Unknown'})
df_crime['Vict Sex'] = df_crime['Vict Sex'].where(
    df_crime['Vict Sex'].isin(['Male', 'Female', 'Unknown']), 'Unknown'
)

# Step 18: Validate LAT/LON; numeric conversion
df_crime['LAT'] = pd.to_numeric(df_crime['LAT'], errors='coerce')
df_crime['LON'] = pd.to_numeric(df_crime['LON'], errors='coerce')

df_crime = df_crime.reset_index(drop=True)
df_crime['row_id'] = df_crime.index


# ── System C variable derivations (from df_crime) ─────────────

# DATE OCC alias — chart code throughout uses "DATE OCC";
# preprocessing stores the parsed date in "Date Rptd"
df_crime["DATE OCC"] = df_crime["Date Rptd"]

# Age Band (System C heatmap uses "<18", "18-25" etc.)
def band_age(age):
    if pd.isna(age): return None
    age = float(age)
    if age < 18:   return "<18"
    if age <= 25:  return "18-25"
    if age <= 40:  return "26-40"
    if age <= 60:  return "41-60"
    return "60+"

df_crime["Age Band"]    = df_crime["Vict Age"].apply(band_age)
df_crime["Crime Group"] = df_crime["crime_category"]

# Map subset: valid LAT/LON within LA bounding box
map_crime = df_crime[
    df_crime["LAT"].notna() & df_crime["LON"].notna() &
    (df_crime["LAT"] != 0)  & (df_crime["LON"] != 0)  &
    df_crime["LAT"].between(33.70, 34.35) &
    df_crime["LON"].between(-118.70, -118.15)
].copy().reset_index(drop=True)

def load_and_prepare(path: Path):
    """Wrapper kept for compatibility — returns already-cleaned globals."""
    return df_crime.copy(), map_crime.copy()

# ── Altair chart spec builders ────────────────────────────────────────────────

def make_temporal_spec(df: pd.DataFrame) -> dict:
    """Area + line chart for monthly crime counts."""
    monthly = (
        df.groupby("MonthStart").size()
        .reset_index(name="count")
        .rename(columns={"MonthStart": "month"})
    )
    monthly["month"] = monthly["month"].dt.strftime("%Y-%m-%d")

    base = alt.Chart(monthly).encode(
        x=alt.X("month:T", title="Month",
                axis=alt.Axis(labelAngle=-30, format="%b %Y", labelOverlap=True)),
        y=alt.Y("count:Q", title="Reported incidents"),
        tooltip=[
            alt.Tooltip("month:T", title="Month", format="%b %Y"),
            alt.Tooltip("count:Q", title="Incidents", format=","),
        ]
    )

    area = base.mark_area(line=True, color="#5b9bd5", opacity=0.55,
                          point=alt.OverlayMarkDef(size=30))

    chart = (
        alt.layer(area)
        .properties(width="container", height=237, background="#fffdf9",
                    title=alt.TitleParams("Monthly reported crime over time",
                                          fontSize=11, anchor="start"))
    )
    return chart.to_dict()


def make_crime_bar_spec(df: pd.DataFrame) -> dict:
    """Horizontal bar — top 12 most frequent crime descriptions."""
    top = (
        df.groupby("Crm Cd Desc").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(12)
        .rename(columns={"Crm Cd Desc": "desc"})
    )

    chart = (
        alt.Chart(top)
        .mark_bar(cursor="pointer", color="#5b9bd5")
        .encode(
            y=alt.Y("desc:N", title="", sort="-x",
                    axis=alt.Axis(labelLimit=210, labelFontSize=11)),
            x=alt.X("count:Q", title="Incident count"),
            tooltip=[
                alt.Tooltip("desc:N",  title="Crime"),
                alt.Tooltip("count:Q", title="Incidents", format=","),
            ]
        )
        .properties(width="container", height=237, background="#fffdf9",
                    title=alt.TitleParams("Most frequent detailed crime descriptions",
                                          fontSize=11, anchor="start"))
    )
    return chart.to_dict()


def make_map_spec(map_df: pd.DataFrame) -> dict:
    """Scatter map — lat/lon points coloured by crime group."""
    plot_df = map_df[["LAT", "LON", "Crm Cd Desc", "AREA NAME", "DATE OCC",
                       "Crime Group"]].copy()
    plot_df["DATE OCC"] = plot_df["DATE OCC"].dt.strftime("%Y-%m-%d")

    chart = (
        alt.Chart(plot_df)
        .mark_point(size=18, opacity=0.48, filled=True, color="#5b9bd5")
        .encode(
            x=alt.X("LON:Q", title="Longitude",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=11)),
            y=alt.Y("LAT:Q", title="Latitude",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(labelFontSize=11)),
            tooltip=[
                alt.Tooltip("Crm Cd Desc:N", title="Crime"),
                alt.Tooltip("AREA NAME:N",   title="Area"),
                alt.Tooltip("DATE OCC:N",    title="Date"),
            ]
        )
        .properties(width="container", height=275, background="#fffdf9",
                    title=alt.TitleParams(
                        f"Selected spatial sample ({len(plot_df):,} points shown)",
                        fontSize=11, anchor="start"))
    )
    return chart.to_dict()


def make_age_heatmap_spec(df: pd.DataFrame) -> dict:
    """Rect heatmap — Age Band × Crime Group."""
    matrix = (
        df.dropna(subset=["Age Band", "Crime Group"])
        .groupby(["Crime Group", "Age Band"]).size()
        .reset_index(name="count")
    )

    chart = (
        alt.Chart(matrix)
        .mark_rect(stroke="#fff", strokeWidth=1, cursor="pointer")
        .encode(
            x=alt.X("Age Band:O", title="Age band",
                    sort=AGE_BANDS,
                    axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("Crime Group:O", title="Crime group",
                    sort=CRIME_GROUP_ORDER),
            color=alt.Color(
                "count:Q",
                title="Victims",
                scale=alt.Scale(
                    domain=[0, matrix["count"].max()],
                    range=[
                        "#4682B4",
                        "#fed976",
                        "#feb24c",
                        "#fd8d3c",
                        "#cc0000"
                    ]
                )
            ),
            tooltip=[
                alt.Tooltip("Crime Group:N", title="Crime group"),
                alt.Tooltip("Age Band:O",    title="Age band"),
                alt.Tooltip("count:Q",       title="Victims", format=","),
            ]
        )
        .properties(width="container", height=192, background="#fffdf9",
                    title=alt.TitleParams("Victim Age-band by crime group",
                                          fontSize=11, anchor="start"))
    )
    return chart.to_dict()


def make_sex_spec(df: pd.DataFrame) -> dict:
    """Stacked 100% bar — Victim sex composition per crime group."""
    SEXES = ["Female", "Male", "Unknown"]
    totals = (
        df.groupby("Crime Group").size()
        .reset_index(name="total")
    )
    sex_data = (
        df[df["Vict Sex"].isin(SEXES)]
        .groupby(["Crime Group", "Vict Sex"]).size()
        .reset_index(name="count")
        .rename(columns={"Vict Sex": "sex"})
        .merge(totals, on="Crime Group")
    )
    sex_data["pct"] = sex_data["count"] / sex_data["total"] * 100

    chart = (
        alt.Chart(sex_data)
        .mark_bar(cursor="pointer")
        .encode(
            x=alt.X("pct:Q", title="Share of victims (%)",
                    axis=alt.Axis(format=".0f", labelExpr="datum.value+'%'"),
                    scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Crime Group:N", title="Crime group",
                    sort=CRIME_GROUP_ORDER,
                    axis=alt.Axis(labelLimit=160)),
            color=alt.Color("sex:N",
                            scale=alt.Scale(
                                domain=["Unknown","Female", "Male"],
                                range=["#2ca02c", "#ff8c1a", "#1f77b4"]
                            ),
                            legend=alt.Legend(orient="top", title=None)),
            order=alt.Order("sex:N", sort="ascending"),
            tooltip=[
                alt.Tooltip("Crime Group:N", title="Crime group"),
                alt.Tooltip("sex:N",         title="Sex"),
                alt.Tooltip("pct:Q",         title="Share (%)", format=".1f"),
                alt.Tooltip("count:Q",       title="Count",     format=","),
            ]
        )
        .properties(width="container", height=192, background="#fffdf9",
                    title=alt.TitleParams("Victim sex composition within each crime group",
                                          fontSize=11, anchor="start"))
    )
    return chart.to_dict()


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(df: pd.DataFrame, map_df: pd.DataFrame) -> str:
    records = df[[
        "row_id", "DATE OCC", "Year", "MonthStart", "Crm Cd Desc",
        "Crime Group", "AREA NAME", "Vict Age", "Age Band", "Vict Sex", "LAT", "LON",
    ]].copy()
    records["DATE OCC"]   = records["DATE OCC"].dt.strftime("%Y-%m-%d")
    records["MonthStart"] = records["MonthStart"].dt.strftime("%Y-%m-%d")
    records = records.replace({np.nan: None})

    map_records = map_df[[
        "row_id", "DATE OCC", "Year", "Crm Cd Desc", "Crime Group",
        "AREA NAME", "Vict Age", "Age Band", "Vict Sex", "LAT", "LON",
    ]].copy()
    map_records["DATE OCC"] = map_records["DATE OCC"].dt.strftime("%Y-%m-%d")
    map_records = map_records.replace({np.nan: None})

    years = sorted(int(y) for y in df["Year"].dropna().unique().tolist())
    crime_groups = [g for g in CRIME_GROUP_ORDER if g in set(df["Crime Group"].dropna())]
    meta = {
        "areas"       : sorted(df["AREA NAME"].dropna().unique().tolist()),
        "crime_groups": crime_groups,
        "year_min"    : int(min(years)),
        "year_max"    : int(max(years)),
    }

    # ── Build Altair specs ────────────────────────────────────────────────────
    temporal_spec_dict  = make_temporal_spec(df)
    crime_bar_spec_dict = make_crime_bar_spec(df)
    map_spec_dict       = make_map_spec(map_df)
    age_heatmap_dict    = make_age_heatmap_spec(df)
    sex_spec_dict       = make_sex_spec(df)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>System 3: Crime Analysis Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
  <style>
    :root {{
      --bg: #e8f4fb; --ink: #202020; --muted: #5a5a5a;
      --panel: #ffffff; --line: #c8dff0;
      --accent: #2a7ab8; --accent-soft: #d0eaf8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Georgia,"Times New Roman",serif; background:var(--bg); color:var(--ink); }}
    .wrap {{ max-width:1000px; margin:0 auto; padding:20px; }}
    .hero,.panel {{ background:var(--panel); border:1px solid var(--line); }}
    .hero  {{ padding:18px; margin-bottom:14px; }}
    .hero {{
        background: #ffffff;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        }}
    .hero h2 {{ margin:0 0 4px; font-size:1.5rem; }}
    .hero h3 {{
            margin: 0;
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            }}
    .hero ul,
    .hero ol {{
        margin: 10px 0 0 18px;
        padding: 0;
        font-size: 14px;
        color: #444;
        }}
    .hero b {{
        color: #0077cc;
        }}
    .hero p  {{ margin:0; font-size:0.88rem; color:var(--muted); }}
    .panel {{ padding:12px; }}

    .muted {{ color:var(--muted); }}
    .grid2 {{ display:grid; grid-template-columns:1.05fr 0.95fr; gap:10px; margin-bottom:10px; }}
    .controls {{ display:grid; grid-template-columns:1.2fr 1fr 1fr auto auto; gap:9px; margin-bottom:10px; }}
    .control {{ background: var(--panel); padding: 9px; }}
    .panel, .control {{ box-shadow: 0 1px 4px rgba(42,122,184,0.10); }}
    .panel-title {{ display:flex; justify-content:space-between; gap:6px; align-items:baseline; margin-bottom:5px; }}
    .chart {{ min-height:275px; width:100%; }}
    .chart-sm {{ min-height:218px; width:100%; }}
    label {{ display:block; font-weight:900; margin-bottom:5px; font-size:1.0rem; }}
    select, button {{ width:100%; padding:6px 8px; border:1px solid #cdbfaa; background:#fff; font-family:inherit; font-size:0.61rem; }}
    button {{ cursor:pointer; }}
    .primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
    .secondary {{ background:#fff; color:var(--ink); }}
    .slider-row {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; align-items:center; }}
    .slider-row input {{ width:100%; accent-color:var(--accent); }}
    .yr-readout {{ display:inline-block; background:var(--accent-soft); padding:2px 5px; font-size:0.58rem; margin-bottom:6px; }}
    #filterBadges {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; min-height:0; }}
    .badge {{
      display:inline-flex; align-items:center; gap:4px;
      background:var(--accent-soft); border:1px solid #cdbfaa;
      padding:2px 6px; font-size:0.53rem; border-radius:2px;
    }}
    .badge .x {{ cursor:pointer; font-weight:700; color:var(--accent); margin-left:2px; }}
    .vega-embed summary {{ display:none !important; }}
    @media (max-width:1100px) {{ .grid2,.controls {{ grid-template-columns:1fr; }} }}
    #vis.panel > div {{ width:100%; }}
    .note {{ margin-top:8px; color:var(--muted); font-size:0.82rem; text-align:center; }}

    .system-nav {{
      display:flex;
      flex-wrap:wrap;
      gap:12px;
      margin-bottom:16px;
    }}
    .nav-btn {{
      display:inline-flex;
      align-items:center;
      justify-content:center;
      min-width:150px;
      padding:10px 18px;
      border-radius:10px;
      border:1px solid #d9c9b2;
      background:linear-gradient(135deg, #ffffff, #fbf5eb);
      color:#5d4330;
      font-size:0.95rem;
      font-weight:700;
      text-decoration:none;
      box-shadow:0 3px 10px rgba(0,0,0,0.06);
      transition:transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
    }}
    .nav-btn:hover {{
      transform:translateY(-2px);
      box-shadow:0 8px 18px rgba(0,0,0,0.10);
      background:linear-gradient(135deg, #ffffff, #fff0cf);
    }}
  </style>
</head>
<body>
<div class="wrap">


  <div class="system-nav">
    <a class="nav-btn" href="dashboard.html">Dashboard</a>
    <a class="nav-btn" href="index.html">System A</a>
    <a class="nav-btn" href="SYSTEM_C.html">System C</a>
  </div>
  
  <section class="hero">
    <h2>System C: Crime Data Analysis</h2>
  </section>

  <section class="hero">
    <h3>Instruction: </h3>
    <ul style="margin-top:10px; line-height:1.6;">
    <li>Use the controls to filter data by year range, area, or crime group.</li>
    <li>The temporal chart shows monthly crime trends; drag to select a time range</li>
    <li>Click bars in the crime type chart to filter by crime description.</li>
    <li>Drag on the map to select incidents in a region.</li>
    <li>Click cells in the age-band heatmap to highlight related crimes.</li>
    <li>Click segments in the sex distribution chart to filter victims by sex.</li>
    <li>All charts update together through linked interactions.</li>
    <li>Click <b>Reset</b> to clear all selections.</li>
    </ul>
  </section>

  <section class="hero">
    <h3>Tasks: </h3>
    <ol style="margin-top:10px; line-height:1.6;">
      <li>Analyse temporal trends (year, month, time of day) to identify patterns, seasonality, and spikes in crime.</li>
      <li>Examine the spatial distribution of crimes to identify geographic patterns and hotspots.</li>
      <li>Investigate crime categories/types and compare their frequencies.</li>
      <li>Analyse victim demographics (age and sex) in relation to different crime types.</li>
      <li>Use interactive filters and selection (brushing) to explore specific subsets of incidents.</li>
      <li>Observe linked/coordinated views to compare selected subsets with the overall dataset.</li>
    </ol>
  </section>

  <section class="controls">
    <div class="control">
      <label>Year range</label>
      <div class="yr-readout" id="yearReadout"></div>
      <div class="slider-row">
        <input id="yearStart" type="range" min="{meta['year_min']}" max="{meta['year_max']}" step="1" value="{meta['year_min']}">
        <input id="yearEnd"   type="range" min="{meta['year_min']}" max="{meta['year_max']}" step="1" value="{meta['year_max']}">
      </div>
    </div>
    <div class="control">
      <label for="areaSelect">Area filter</label>
      <select id="areaSelect"></select>
    </div>
    <div class="control">
      <label for="crimeGroupSelect">Crime group filter</label>
      <select id="crimeGroupSelect"></select>
    </div>
    <div class="control">
      <label>&nbsp;</label>
      <button id="generaliseBtn" class="secondary">Generalise selection</button>
    </div>
    <div class="control">
      <label>&nbsp;</label>
      <button id="resetBtn" class="primary">Reset all interactions</button>
    </div>
  </section>

  <div id="filterBadges"></div>

  <div class="grid2">
    <section class="panel">
      <div class="panel-title"><h3>Temporal overview of crime</h3></div>
      <div id="temporal" class="chart"></div>
    </section>
    <section class="panel">
      <div class="panel-title"><h3>Crime type ranking</h3></div>
      <div id="crimeBar" class="chart"></div>
    </section>
  </div>

  <section class="panel" style="margin-bottom:10px;">
    <div class="panel-title"><h2>Spatial distribution</h2></div>
    <div id="mapChart" class="chart"></div>
  </section>

  <div class="grid2">
    <section class="panel">
      <div class="panel-title"><h2>Age band × crime group heatmap</h2></div>
      <div id="ageHeatmap" class="chart-sm"></div>
    </section>
    <section class="panel">
      <div class="panel-title"><h2>Victim sex composition</h2></div>
      <div id="sexPercent" class="chart-sm"></div>
    </section>
  </div>

</div>

<script>
// ═══════════════════════════════════════════════════════════
//  DATA  (injected by Python / Altair pipeline)
// ═══════════════════════════════════════════════════════════
const DATA     = {json.dumps(records.to_dict(orient='records'))};
const MAP_DATA = {json.dumps(map_records.to_dict(orient='records'))};
const META     = {json.dumps(meta)};
const CRIME_GROUP_ORDER = {json.dumps(crime_groups)};
const AGE_BANDS = {json.dumps(AGE_BANDS)};
const SEXES = ["Female","Male","Unknown"];

// Altair-generated base specs
const BASE_SPECS = {{
  temporal  : {json.dumps(temporal_spec_dict)},
  crimeBar  : {json.dumps(crime_bar_spec_dict)},
  map       : {json.dumps(map_spec_dict)},
  ageHeatmap: {json.dumps(age_heatmap_dict)},
  sex       : {json.dumps(sex_spec_dict)},
}};

const DESC_TO_GROUP = {{}};
DATA.forEach(d => {{ if (d["Crm Cd Desc"] && d["Crime Group"]) DESC_TO_GROUP[d["Crm Cd Desc"]] = d["Crime Group"]; }});

// ═══════════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════════
const state = {{
  yearStart     : META.year_min,
  yearEnd       : META.year_max,
  area          : "All",
  crimeGroup    : "All",
  selectedDescs : new Set(),
  dateRange     : null,
  mapIds        : new Set(),
  heatmapCell   : null,
  sexCell       : null,
}};

let temporalView=null, crimeBarView=null, mapView=null,
    ageHeatmapView=null, sexView=null;

// ═══════════════════════════════════════════════════════════
//  SELECTS
// ═══════════════════════════════════════════════════════════
function fillSelect(id, values, allLabel) {{
  const sel = document.getElementById(id);
  sel.innerHTML = "";
  const all = document.createElement("option");
  all.value = "All"; all.textContent = allLabel;
  sel.appendChild(all);
  values.forEach(v => {{
    const o = document.createElement("option");
    o.value = String(v); o.textContent = String(v);
    sel.appendChild(o);
  }});
}}
fillSelect("areaSelect",       META.areas,        "All areas");
fillSelect("crimeGroupSelect", META.crime_groups, "All crime groups");

function updateYearReadout() {{
  if (state.yearStart > state.yearEnd) {{
    [state.yearStart, state.yearEnd] = [state.yearEnd, state.yearStart];
    document.getElementById("yearStart").value = state.yearStart;
    document.getElementById("yearEnd").value   = state.yearEnd;
  }}
  document.getElementById("yearReadout").textContent =
    `Showing years ${{state.yearStart}} to ${{state.yearEnd}}`;
}}

// ═══════════════════════════════════════════════════════════
//  FILTER
// ═══════════════════════════════════════════════════════════
function passesStateFilters(d) {{
  if (d.Year < state.yearStart || d.Year > state.yearEnd) return false;
  if (state.area !== "All" && d["AREA NAME"] !== state.area) return false;
  if (state.crimeGroup !== "All" && d["Crime Group"] !== state.crimeGroup) return false;
  if (state.selectedDescs.size && !state.selectedDescs.has(d["Crm Cd Desc"])) return false;
  if (state.dateRange) {{
    if (d["DATE OCC"] < state.dateRange[0] || d["DATE OCC"] > state.dateRange[1]) return false;
  }}
  if (state.mapIds.size && !state.mapIds.has(d.row_id)) return false;
  return true;
}}

function filteredRows()    {{ return DATA.filter(passesStateFilters); }}
function filteredMapRows() {{ return MAP_DATA.filter(passesStateFilters); }}

// ═══════════════════════════════════════════════════════════
//  SUMMARY BUILDERS  (JS-side, mirrors Altair logic)
// ═══════════════════════════════════════════════════════════
function buildTemporalData(rows) {{
  const mc = {{}};
  rows.forEach(d => {{ if (d.MonthStart) mc[d.MonthStart] = (mc[d.MonthStart]||0)+1; }});
  return Object.keys(mc).sort().map(k => ({{ month:k, count:mc[k] }}));
}}

function buildCrimeBarData(rows) {{
  const cc = {{}};
  rows.forEach(d => {{ if (d["Crm Cd Desc"]) cc[d["Crm Cd Desc"]] = (cc[d["Crm Cd Desc"]]||0)+1; }});
  return Object.entries(cc)
    .sort((a,b)=>b[1]-a[1]).slice(0,12)
    .map(([desc,count])=>({{ desc, count, selected: state.selectedDescs.has(desc) ? 1 : 0 }}));
}}

function buildAgeHeatmapData(rows) {{
  const matrix = {{}};
  CRIME_GROUP_ORDER.forEach(g => AGE_BANDS.forEach(a => {{ matrix[g+"||"+a]=0; }}));
  rows.forEach(d => {{
    const key = d["Crime Group"]+"||"+d["Age Band"];
    if (key in matrix) matrix[key]++;
  }});
  return Object.entries(matrix).map(([k,v]) => {{
    const [crimeGroup,ageBand] = k.split("||");
    const dimmed = state.sexCell && state.sexCell.crimeGroup !== crimeGroup ? 1 : 0;
    const highlighted = state.heatmapCell &&
      state.heatmapCell.crimeGroup===crimeGroup &&
      state.heatmapCell.ageBand===ageBand ? 1 : 0;
    return {{ crimeGroup, ageBand, count:v, dimmed, highlighted }};
  }});
}}

function buildSexData(rows) {{
  const matrix = {{}};
  CRIME_GROUP_ORDER.forEach(g => SEXES.forEach(s => {{ matrix[g+"||"+s]=0; }}));
  rows.forEach(d => {{
    const sex = SEXES.includes(d["Vict Sex"]) ? d["Vict Sex"] : "Unknown";
    const key = d["Crime Group"]+"||"+sex;
    if (key in matrix) matrix[key]++;
  }});
  const totals = {{}};
  CRIME_GROUP_ORDER.forEach(g => {{
    totals[g] = SEXES.reduce((s,sx)=>s+matrix[g+"||"+sx],0);
  }});
  return Object.entries(matrix).map(([k,cnt]) => {{
    const [crimeGroup,sex] = k.split("||");
    const total = totals[crimeGroup]||0;
    const dimmed = state.heatmapCell && state.heatmapCell.crimeGroup !== crimeGroup ? 1 : 0;
    const highlighted = state.sexCell &&
      state.sexCell.crimeGroup===crimeGroup &&
      state.sexCell.sex===sex ? 1 : 0;
    return {{ crimeGroup, sex, count:cnt, pct: total?(cnt/total)*100:0, dimmed, highlighted }};
  }});
}}

// ═══════════════════════════════════════════════════════════
//  SPEC CLONERS  — clone Altair base specs, inject fresh data
// ═══════════════════════════════════════════════════════════

function temporalSpec(data) {{
  const spec = JSON.parse(JSON.stringify(BASE_SPECS.temporal));
  spec.layer[0].data = {{ values: data }};
  // add brush param
  spec.layer[0].params = [{{
    name:"timeBrush",
    select:{{ type:"interval", encodings:["x"], translate:true, zoom:false }}
  }}];
  spec.layer[0].encoding.opacity = {{
    condition:{{ param:"timeBrush", value:1.0 }}, value:0.3
  }};
  return spec;
}}

function crimeBarSpec(data) {{
  const spec = JSON.parse(JSON.stringify(BASE_SPECS.crimeBar));
  spec.data = {{ values: data }};
  spec.params = [{{ name:"barSelect", select:{{ type:"point", fields:["desc"], toggle:false }} }}];
  spec.encoding.color = {{
    condition:{{ param:"barSelect", value:"#1a4f7a" }}, value:"#5b9bd5"
  }};
  spec.encoding.opacity = {{
    condition:{{ param:"barSelect", value:1.0 }}, value:0.45
  }};
  return spec;
}}

function mapSpec(data) {{
  const spec = JSON.parse(JSON.stringify(BASE_SPECS.map));
  spec.data = {{ values: data }};
  spec.title.text = `Spatial sample (${{data.length.toLocaleString()}} pts) — drag to brush region`;
  // Interval brush: selecting a region drives all other charts
  spec.params = [];
  return spec;
}}

function ageHeatmapSpec(data) {{
  const spec = JSON.parse(JSON.stringify(BASE_SPECS.ageHeatmap));
  spec.data = {{ values: data }};
  spec.params = [{{ name:"heatSel", select:{{ type:"point", fields:["crimeGroup","ageBand"] }} }}];
  // cross-highlight opacity
  spec.encoding.opacity = {{
    condition:[
      {{ test:"datum.dimmed === 1", value:0.2 }},
      {{ test:"datum.highlighted === 1", value:1.0 }}
    ],
    value:0.85
  }};
  spec.encoding.strokeWidth = {{
    condition:{{ param:"heatSel", value:2.5 }}, value:0.5
  }};
  // remap fields from Altair names
  spec.encoding.x.field = "ageBand";
  spec.encoding.y.field = "crimeGroup";
  return spec;
}}

function sexSpec(data) {{
  const spec = JSON.parse(JSON.stringify(BASE_SPECS.sex));
  spec.data = {{ values: data }};
  spec.params = [{{ name:"sexSel", select:{{ type:"point", fields:["crimeGroup","sex"] }} }}];
  spec.encoding.opacity = {{
    condition:[
      {{ test:"datum.dimmed === 1",      value:0.2 }},
      {{ test:"datum.highlighted === 1", value:1.0 }}
    ],
    value:0.85
  }};
  spec.encoding.strokeWidth = {{ value:0 }};
  spec.encoding.stroke      = {{ value:"transparent" }};
  // remap fields from Altair names
  spec.encoding.x.field = "pct";
  spec.encoding.y.field = "crimeGroup";
  spec.encoding.color.field = "sex";
  return spec;
}}

// ═══════════════════════════════════════════════════════════
//  EMBED + WIRE LISTENERS
// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
//  MULTIDIRECTIONAL LINKING — central dispatcher
//  Every chart interaction calls reembedAll(), which re-renders
//  ALL charts filtered by the updated state.  No chart is
//  passive: Temporal ↔ CrimeBar ↔ Map ↔ AgeHeatmap ↔ SexBars
// ═══════════════════════════════════════════════════════════

// skipId: the chart that just fired — skip re-embedding it to
// avoid an infinite listener loop (pass null to refresh all 5).
function reembedAll(skipId, opts) {{
  const rows = filteredRows();
  const mr   = filteredMapRows();
  if (skipId !== "#temporal")
    vegaEmbed("#temporal",  temporalSpec(buildTemporalData(rows)),     opts)
      .then(r => {{ temporalView  = r.view; wireTemporalListener(opts); }});
  if (skipId !== "#crimeBar")
    vegaEmbed("#crimeBar",  crimeBarSpec(buildCrimeBarData(rows)),     opts)
      .then(r => {{ crimeBarView  = r.view; wireCrimeBarListener(opts); }});
  if (skipId !== "#mapChart")
    vegaEmbed("#mapChart",  mapSpec(mr),                               opts)
      .then(r => {{ mapView       = r.view; wireMapListener(opts); }});
  if (skipId !== "#ageHeatmap")
    vegaEmbed("#ageHeatmap",ageHeatmapSpec(buildAgeHeatmapData(rows)), opts)
      .then(r => {{ ageHeatmapView= r.view; wireHeatmapListener(opts); }});
  if (skipId !== "#sexPercent")
    vegaEmbed("#sexPercent",sexSpec(buildSexData(rows)),               opts)
      .then(r => {{ sexView       = r.view; wireSexListener(opts); }});
  updateBadges();
}}

// ── per-chart listener wires ────────────────────────────────

function wireTemporalListener(opts) {{
  temporalView.addSignalListener("timeBrush", (_, value) => {{
    if (!value || !value.month || value.month.length < 2) {{
      state.dateRange = null;
    }} else {{
      const fmt = ts => new Date(ts).toISOString().slice(0,10);
      state.dateRange = [fmt(value.month[0]), fmt(value.month[1])];
    }}
    reembedAll("#temporal", opts);
  }});
}}

function wireCrimeBarListener(opts) {{
  crimeBarView.addSignalListener("barSelect", (_, value) => {{
    if (value && value.desc && value.desc.length) {{
      const s = new Set(); value.desc.forEach(d => s.add(d)); state.selectedDescs = s;
    }} else {{
      state.selectedDescs = new Set();
    }}
    updateGeneraliseInfo();
    reembedAll("#crimeBar", opts);
  }});
}}

function wireMapListener(opts) {{
  mapView.addSignalListener("mapBrush", (_, value) => {{
    if (value && value.x && value.x.length === 2 && value.y && value.y.length === 2) {{
      const [lon0, lon1] = value.x, [lat0, lat1] = value.y;
      const ids = new Set();
      MAP_DATA.forEach(d => {{
        if (passesStateFilters(d) &&
            d.LON >= lon0 && d.LON <= lon1 &&
            d.LAT >= lat0 && d.LAT <= lat1) ids.add(d.row_id);
      }});
      state.mapIds = ids;
    }} else {{
      state.mapIds = new Set();
    }}
    // Re-embed ALL including map (opacity needs refresh after brush)
    scheduleRefresh();
  }});
}}

function wireHeatmapListener(opts) {{
  ageHeatmapView.addSignalListener("heatSel", (_, value) => {{
    if (value && value.crimeGroup && value.crimeGroup.length) {{
      const cell = {{ crimeGroup:value.crimeGroup[0], ageBand:value.ageBand[0] }};
      const same = state.heatmapCell &&
        state.heatmapCell.crimeGroup===cell.crimeGroup &&
        state.heatmapCell.ageBand===cell.ageBand;
      state.heatmapCell = same ? null : cell;
    }} else {{
      state.heatmapCell = null;
    }}
    // Heatmap cell selection spotlights matching map points
    if (state.heatmapCell) {{
      const ids = new Set();
      MAP_DATA.forEach(d => {{
        if (d["Crime Group"]===state.heatmapCell.crimeGroup &&
            d["Age Band"]   ===state.heatmapCell.ageBand) ids.add(d.row_id);
      }});
      state.mapIds = ids;
    }} else {{
      state.mapIds = new Set();
    }}
    reembedAll("#ageHeatmap", opts);
  }});
}}

function wireSexListener(opts) {{
  sexView.addSignalListener("sexSel", (_, value) => {{
    if (value && value.crimeGroup && value.crimeGroup.length) {{
      const cell = {{ crimeGroup:value.crimeGroup[0], sex:value.sex[0] }};
      const same = state.sexCell &&
        state.sexCell.crimeGroup===cell.crimeGroup &&
        state.sexCell.sex===cell.sex;
      state.sexCell = same ? null : cell;
    }} else {{
      state.sexCell = null;
    }}
    reembedAll("#sexPercent", opts);
  }});
}}

// ── initial bootstrap (called by refresh()) ─────────────────

function embedAll(subsetRows, subsetMapRows) {{
  const opts = {{ actions:false, renderer:"canvas" }};
  vegaEmbed("#temporal",  temporalSpec(buildTemporalData(subsetRows)),     opts)
    .then(r => {{ temporalView  = r.view; wireTemporalListener(opts); }});
  vegaEmbed("#crimeBar",  crimeBarSpec(buildCrimeBarData(subsetRows)),     opts)
    .then(r => {{ crimeBarView  = r.view; wireCrimeBarListener(opts); }});
  vegaEmbed("#mapChart",  mapSpec(subsetMapRows),                          opts)
    .then(r => {{ mapView       = r.view; wireMapListener(opts); }});
  vegaEmbed("#ageHeatmap",ageHeatmapSpec(buildAgeHeatmapData(subsetRows)), opts)
    .then(r => {{ ageHeatmapView= r.view; wireHeatmapListener(opts); }});
  vegaEmbed("#sexPercent",sexSpec(buildSexData(subsetRows)),               opts)
    .then(r => {{ sexView       = r.view; wireSexListener(opts); }});
}}

// ═══════════════════════════════════════════════════════════
//  BADGES
// ═══════════════════════════════════════════════════════════
function updateBadges() {{
  const wrap = document.getElementById("filterBadges");
  wrap.innerHTML = "";
  const add = (label, clearFn) => {{
    const b = document.createElement("span");
    b.className = "badge";
    b.innerHTML = `${{label}} <span class="x" title="Clear">✕</span>`;
    b.querySelector(".x").addEventListener("click", clearFn);
    wrap.appendChild(b);
  }};
  if (state.area !== "All")
    add(`Area: ${{state.area}}`, () => {{ state.area="All"; document.getElementById("areaSelect").value="All"; scheduleRefresh(); }});
  if (state.crimeGroup !== "All")
    add(`Group: ${{state.crimeGroup}}`, () => {{ state.crimeGroup="All"; document.getElementById("crimeGroupSelect").value="All"; scheduleRefresh(); }});
  if (state.selectedDescs.size) {{
    const label = state.selectedDescs.size===1
      ? `Crime: ${{[...state.selectedDescs][0].slice(0,30)}}${{[...state.selectedDescs][0].length>30?"…":""}}`
      : `Crimes: ${{state.selectedDescs.size}} selected`;
    add(label, () => {{ state.selectedDescs=new Set(); updateGeneraliseInfo(); scheduleRefresh(); }});
  }}
  if (state.dateRange)
    add(`Date: ${{state.dateRange[0]}} – ${{state.dateRange[1]}}`, () => {{ state.dateRange=null; scheduleRefresh(); }});
  if (state.mapIds.size && !state.heatmapCell)
    add(`Map brush: ${{state.mapIds.size}} pts`,
      () => {{ state.mapIds=new Set(); scheduleRefresh(); }});
  if (state.heatmapCell)
    add(`Age cell: ${{state.heatmapCell.crimeGroup}} / ${{state.heatmapCell.ageBand}} (${{state.mapIds.size}} map pts)`,
      () => {{ state.heatmapCell=null; state.mapIds=new Set(); scheduleRefresh(); }});
  if (state.sexCell)
    add(`Sex cell: ${{state.sexCell.crimeGroup}} / ${{state.sexCell.sex}}`,
      () => {{ state.sexCell=null; scheduleRefresh(); }});
}}

// ═══════════════════════════════════════════════════════════
//  GENERALISE FEATURE
// ═══════════════════════════════════════════════════════════
function updateGeneraliseInfo() {{
  // (kept minimal — no panel element needed for core logic)
}}

function generaliseSelection() {{
  if (!state.selectedDescs.size) {{
    window.alert("First select a crime description from the crime ranking chart.");
    return;
  }}
  const firstDesc   = [...state.selectedDescs][0];
  const parentGroup = DESC_TO_GROUP[firstDesc];
  if (!parentGroup) {{ window.alert("No broader crime group found."); return; }}
  state.selectedDescs = new Set();
  state.crimeGroup = parentGroup;
  document.getElementById("crimeGroupSelect").value = parentGroup;
  refresh();
}}

// ═══════════════════════════════════════════════════════════
//  REFRESH
// ═══════════════════════════════════════════════════════════
function refresh() {{
  updateYearReadout();
  const rows    = filteredRows();
  const mapRows = filteredMapRows();
  embedAll(rows, mapRows);
  updateBadges();
}}

let refreshTimer = null;
function scheduleRefresh(delay=60) {{
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => {{ refreshTimer=null; refresh(); }}, delay);
}}

// ═══════════════════════════════════════════════════════════
//  CONTROL LISTENERS
// ═══════════════════════════════════════════════════════════
document.getElementById("yearStart").addEventListener("input", e => {{ state.yearStart = Number(e.target.value); scheduleRefresh(); }});
document.getElementById("yearEnd").addEventListener("input",   e => {{ state.yearEnd   = Number(e.target.value); scheduleRefresh(); }});
document.getElementById("areaSelect").addEventListener("change",      e => {{ state.area = e.target.value; scheduleRefresh(); }});
document.getElementById("crimeGroupSelect").addEventListener("change", e => {{ state.crimeGroup = e.target.value; state.selectedDescs = new Set(); scheduleRefresh(); }});
document.getElementById("generaliseBtn").addEventListener("click", generaliseSelection);
document.getElementById("resetBtn").addEventListener("click", () => {{
  state.yearStart = META.year_min; state.yearEnd = META.year_max;
  state.area = "All"; state.crimeGroup = "All";
  state.mapIds = new Set(); state.selectedDescs = new Set();
  state.dateRange = null; state.heatmapCell = null; state.sexCell = null;
  document.getElementById("yearStart").value = META.year_min;
  document.getElementById("yearEnd").value   = META.year_max;
  document.getElementById("areaSelect").value       = "All";
  document.getElementById("crimeGroupSelect").value = "All";
  refresh();
}});

scheduleRefresh(0);
</script>
</body>
</html>'''


def main() -> None:
    df, map_df = load_and_prepare(DATA_PATH)
    html_text = build_html(df, map_df)
    OUTPUT_HTML.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML.resolve()} from {len(df):,} cleaned records.")


if __name__ == "__main__":
    main()
