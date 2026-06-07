import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd

alt.data_transformers.disable_max_rows()
alt.renderers.enable("default")

# ── Paths ─────────────────────────────────────────────────────
file_path   = "Crime_Data_from_2020_to_Present.csv"
output_file = "SYSTEM_B.html"

# ── Month ordering (temporal heatmap x-axis) ──────────────────
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Age bin config (victim age chart) ─────────────────────────
age_bins   = list(range(0, 101, 10))
age_labels = [f"{i}-{i+9}" for i in range(0, 100, 10)]

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

# =========================================================
# END OF PREPROCESSING — df_crime is now fully cleaned
# =========================================================

# ── System B variable derivations (from df_crime) ─────────────

# Age group bins (System B chart expects "Vict Age Group" column)
df_crime["Vict Age Group"] = pd.cut(
    df_crime["Vict Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
    right=False,
)

# month_name for temporal heatmap y-axis
df_crime["month_name"] = df_crime["Date Rptd"].dt.strftime("%b")

# Keep top-15 crime descriptions for cleaner dashboard
TOP_N      = 15
top_crimes = df_crime["Crm Cd Desc"].value_counts().head(TOP_N).index.tolist()
crime_top  = df_crime[df_crime["Crm Cd Desc"].isin(top_crimes)].copy()

# Map subset: valid LAT/LON within LA bounding box
map_df = df_crime[
    df_crime["LAT"].notna() & df_crime["LON"].notna() &
    (df_crime["LAT"] != 0)  & (df_crime["LON"] != 0)  &
    df_crime["LAT"].between(33.70, 34.35) &
    df_crime["LON"].between(-118.70, -118.15)
].copy()
map_df = map_df[map_df["Crm Cd Desc"].isin(top_crimes)].copy()
if len(map_df) > 5000:
    map_df = map_df.sample(n=5000, random_state=42).copy()

map_df["Vict Age Group"] = pd.cut(
    map_df["Vict Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True,
    right=False,
)
map_df["month_name"] = map_df["Date Rptd"].dt.strftime("%b")
map_df = map_df.reset_index(drop=True)

# -----------------------------
# Theme colors
# -----------------------------
BASE_GRAY = "#c7c7c7"
BRUSH_BLUE = "#2a6fdb"
INCIDENT_BLUE = "#0b3c5d"
SEX_TEAL = "#1b9e77"
CRIME_ORANGE = "#e67e22"
AREA_GOLD = "#f4b942"
PANEL_BG = "#fffdf9"
PAGE_BG = "#e8f4fb"
LINE = "#ded7cc"
ACCENT = "#7a5c43"

# -----------------------------
# Interactive controls from SYSTEM_B_updated.py
# -----------------------------
brush = alt.selection_interval(name="brush")
incident_pick = alt.selection_point(fields=["DR_NO"], name="incident_pick", on="click", empty=True, clear="dblclick")
area_pick = alt.selection_point(fields=["AREA NAME"], name="area_pick", on="click", empty=True, clear="dblclick")
sex_pick = alt.selection_point(fields=["Vict Sex"], name="sex_pick", on="click", empty=True, clear="dblclick")
crime_pick = alt.selection_point(fields=["Crm Cd Desc"], name="crime_pick", on="click", empty=True, clear="dblclick")

# ── Generalised selection ─────────────────────────────────────────────────────

generalised_area = alt.param(name="generalised_area", value=False)

no_incident_selected = "length(data('incident_pick_store')) == 0"
incident_selected    = "length(data('incident_pick_store')) > 0"
no_area_selected     = "length(data('area_pick_store')) == 0"
area_selected        = "length(data('area_pick_store')) > 0"
no_brush_selected    = "length(data('brush_store')) == 0"
brush_selected       = "length(data('brush_store')) > 0"
no_crime_selected    = "length(data('crime_pick_store')) == 0"
crime_selected       = "length(data('crime_pick_store')) > 0"
no_sex_selected      = "length(data('sex_pick_store')) == 0"
sex_selected         = "length(data('sex_pick_store')) > 0"

# -----------------------------
# Map panel (same function logic, slightly tuned sizing)
# -----------------------------
point_map_base = (
    alt.Chart(map_df)
    .mark_circle(size=26, color=BASE_GRAY)
    .encode(
        x=alt.X("LON:Q", title="Longitude", scale=alt.Scale(zero=False)),
        y=alt.Y("LAT:Q", title="Latitude", scale=alt.Scale(zero=False)),
        opacity=alt.condition(brush, alt.value(0.85), alt.value(0.22)),
        tooltip=[
            alt.Tooltip("DR_NO:N", title="Incident ID"),
            alt.Tooltip("Crm Cd Desc:N", title="Crime Type"),
            alt.Tooltip("AREA NAME:N", title="Area"),
            alt.Tooltip("DATE OCC:T", title="Date"),
            alt.Tooltip("Vict Age:Q", title="Victim Age"),
            alt.Tooltip("Vict Sex:N", title="Victim Sex"),
        ],
    )
)

point_map_sex = (
    alt.Chart(map_df)
    .transform_filter(sex_selected)
    .transform_filter(sex_pick)
    .mark_circle(size=30, color=SEX_TEAL)
    .encode(x="LON:Q", y="LAT:Q")
)

point_map_crime = (
    alt.Chart(map_df)
    .transform_filter(crime_selected)
    .transform_filter(crime_pick)
    .mark_circle(size=30, color=CRIME_ORANGE)
    .encode(x="LON:Q", y="LAT:Q")
)

point_map_incident = (
    alt.Chart(map_df)
    .transform_filter(no_area_selected)
    .transform_filter(incident_pick)
    .mark_circle(size=40, color=INCIDENT_BLUE)
    .encode(x="LON:Q", y="LAT:Q")
)

point_map_area_click = (
    alt.Chart(map_df)
    .transform_filter("!generalised_area")
    .transform_filter(area_selected)
    .transform_filter(area_pick)
    .mark_circle(size=34, color=AREA_GOLD)
    .encode(x="LON:Q", y="LAT:Q")
)

point_map_area_generalised = (
    alt.Chart(map_df)
    .transform_filter("generalised_area")
    .transform_filter(area_selected)
    .transform_filter(area_pick)
    .mark_circle(size=34, color=AREA_GOLD)
    .encode(x="LON:Q", y="LAT:Q")
)

point_map = (
    alt.layer(
        point_map_base,
        point_map_sex,
        point_map_crime,
        point_map_incident,
        point_map_area_click,
        point_map_area_generalised,
    )
    .add_params(brush, incident_pick, generalised_area)
    .properties(width=390, height=250, title="Interactive Crime Incident Map")
)

# -----------------------------
# Temporal heatmap (same logic)
# -----------------------------
def time_layer(base, color=None, scale=None, tooltips=None):
    chart = (
        base
        .mark_rect(stroke="#ffffff", strokeWidth=0.35)
        .encode(
            x=alt.X("month_name:N", sort=month_order, title="Month"),
            y=alt.Y("year:O", title="Year"),
            tooltip=tooltips or [
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("month_name:N", title="Month"),
                alt.Tooltip("count():Q", title="Count"),
            ],
        )
    )
    if color is not None:
        chart = chart.encode(color=alt.value(color))
    else:
        chart = chart.encode(
            color=alt.Color(
                "count():Q",
                title="Crime Count",
                scale=scale or alt.Scale(scheme="blues"),
            )
        )
    return chart

heatmap_base = alt.Chart(crime_top)
heatmap_map_base = alt.Chart(map_df)

time_chart = alt.layer(
    time_layer(
        heatmap_base
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(no_sex_selected)
        .transform_filter(no_brush_selected),
        scale=alt.Scale(scheme="blues"),
    ),
    time_layer(
        heatmap_map_base
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(no_sex_selected)
        .transform_filter(brush_selected)
        .transform_filter(brush),
        scale=alt.Scale(scheme="blues"),
    ),
    time_layer(
        heatmap_map_base
        .transform_filter("!generalised_area")
        .transform_filter(incident_selected)
        .transform_filter(incident_pick),
        color=INCIDENT_BLUE,
        tooltips=[alt.Tooltip("DATE OCC:T", title="Date"), alt.Tooltip("DR_NO:N", title="Incident ID")],
    ),
    time_layer(
        heatmap_base
        .transform_filter("generalised_area")
        .transform_filter(area_selected)
        .transform_filter(area_pick),
        scale=alt.Scale(scheme="goldorange"),
        tooltips=[
            alt.Tooltip("AREA NAME:N", title="Area"),
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("count():Q", title="Count"),
        ],
    ),
    time_layer(
        heatmap_base
        .transform_filter("!generalised_area")
        .transform_filter(crime_selected)
        .transform_filter(crime_pick),
        scale=alt.Scale(scheme="orangered"),
        tooltips=[
            alt.Tooltip("Crm Cd Desc:N", title="Crime Type"),
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("count():Q", title="Count"),
        ],
    ),
    time_layer(
        heatmap_base
        .transform_filter("!generalised_area")
        .transform_filter(no_crime_selected)
        .transform_filter(sex_selected)
        .transform_filter(sex_pick),
        scale=alt.Scale(scheme="teals"),
        tooltips=[
            alt.Tooltip("Vict Sex:N", title="Victim Sex"),
            alt.Tooltip("year:O", title="Year"),
            alt.Tooltip("month_name:N", title="Month"),
            alt.Tooltip("count():Q", title="Count"),
        ],
    ),
).properties(width=300, height=220, title="Temporal Heatmap")

# -----------------------------
# Victim sex chart (same function logic)
# -----------------------------
sex_sort = ["Female", "Male", "Unknown"]

def victim_sex_layer(base, color_value=None, conditional=False):
    chart = (
        base
        .transform_filter("datum['Vict Sex'] != null")
        .transform_aggregate(record_count="count()", groupby=["Vict Sex"])
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Vict Sex:N", title="Victim Sex", sort=sex_sort, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("record_count:Q", title="Record Count"),
            tooltip=[
                alt.Tooltip("Vict Sex:N", title="Victim Sex"),
                alt.Tooltip("record_count:Q", title="Count"),
            ],
        )
    )
    if conditional:
        chart = chart.encode(color=alt.condition(sex_pick, alt.value(SEX_TEAL), alt.value(BASE_GRAY), empty=False))
    else:
        chart = chart.encode(color=alt.value(color_value))
    return chart

victim_chart = alt.layer(
    victim_sex_layer(
        alt.Chart(map_df)
        .add_params(sex_pick)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(no_brush_selected),
        conditional=True,
    ),
    victim_sex_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(brush_selected)
        .transform_filter(brush),
        color_value=BRUSH_BLUE,
    ),
    victim_sex_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(incident_selected)
        .transform_filter(incident_pick),
        color_value=INCIDENT_BLUE,
    ),
    victim_sex_layer(
        alt.Chart(map_df)
        .transform_filter("generalised_area")
        .transform_filter(area_selected)
        .transform_filter(area_pick),
        color_value=AREA_GOLD,
    ),
    victim_sex_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(crime_selected)
        .transform_filter(crime_pick),
        color_value=CRIME_ORANGE,
    ),
).properties(width=232, height=184, title="Victim Sex Distribution")

# -----------------------------
# Victim age chart (same function logic)
# -----------------------------
age_sort = age_labels

def make_age_layer(base_chart, color_value):
    return (
        base_chart
        .transform_filter("datum['Vict Age Group'] != null")
        .transform_aggregate(age_count="count()", groupby=["Vict Age Group"])
        .mark_bar(color=color_value, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Vict Age Group:N", title="Victim Age Group", sort=age_sort, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("age_count:Q", title="Count"),
            tooltip=[
                alt.Tooltip("Vict Age Group:N", title="Age Group"),
                alt.Tooltip("age_count:Q", title="Count"),
            ],
        )
    )

victim_age_chart = alt.layer(
    make_age_layer(
        alt.Chart(crime_top)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(no_sex_selected)
        .transform_filter(no_brush_selected),
        BASE_GRAY,
    ),
    make_age_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_crime_selected)
        .transform_filter(no_sex_selected)
        .transform_filter(brush_selected)
        .transform_filter(brush),
        BRUSH_BLUE,
    ),
    make_age_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(incident_selected)
        .transform_filter(incident_pick),
        INCIDENT_BLUE,
    ),
    make_age_layer(
        alt.Chart(crime_top)
        .transform_filter("generalised_area")
        .transform_filter(area_selected)
        .transform_filter(area_pick),
        AREA_GOLD,
    ),
    make_age_layer(
        alt.Chart(crime_top)
        .transform_filter("!generalised_area")
        .transform_filter(crime_selected)
        .transform_filter(crime_pick),
        CRIME_ORANGE,
    ),
    make_age_layer(
        alt.Chart(crime_top)
        .transform_filter("!generalised_area")
        .transform_filter(no_crime_selected)
        .transform_filter(sex_selected)
        .transform_filter(sex_pick),
        SEX_TEAL,
    ),
).properties(width=232, height=184, title="Victim Age Distribution")

# -----------------------------
# Crime category dot plot (same function logic)
# -----------------------------
def crime_layer(base, color_value=None, conditional=False):
    chart = (
        base
        .transform_aggregate(crime_count="count()", groupby=["Crm Cd Desc"])
        .transform_window(rank="rank(crime_count)", sort=[alt.SortField("crime_count", order="descending")])
        .transform_filter(alt.datum.rank <= 12)
        .mark_circle(size=160)
        .encode(
            x=alt.X("crime_count:Q", title="Incident Count"),
            y=alt.Y("Crm Cd Desc:N", sort="-x", title="Crime Type"),
            tooltip=[
                alt.Tooltip("Crm Cd Desc:N", title="Crime Type"),
                alt.Tooltip("crime_count:Q", title="Count"),
            ],
        )
    )
    if conditional:
        chart = chart.encode(color=alt.condition(crime_pick, alt.value(CRIME_ORANGE), alt.value(BASE_GRAY), empty=False))
    else:
        chart = chart.encode(color=alt.value(color_value))
    return chart

crime_category_chart = alt.layer(
    crime_layer(
        alt.Chart(map_df)
        .add_params(crime_pick)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_brush_selected)
        .transform_filter(no_sex_selected),
        conditional=True,
    ),
    crime_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_sex_selected)
        .transform_filter(brush_selected)
        .transform_filter(brush),
        color_value=BRUSH_BLUE,
    ),
    crime_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(incident_selected)
        .transform_filter(incident_pick),
        color_value=INCIDENT_BLUE,
    ),
    crime_layer(
        alt.Chart(map_df)
        .transform_filter("generalised_area")
        .transform_filter(area_selected)
        .transform_filter(area_pick),
        color_value=AREA_GOLD,
    ),
    crime_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(sex_selected)
        .transform_filter(sex_pick),
        color_value=SEX_TEAL,
    ),
).properties(width=352, height=184, title="Crime Type Dot Plot")

# -----------------------------
# Area chart (same function logic)
# -----------------------------
def area_layer(base, color_value=None, conditional=False):
    chart = (
        base
        .transform_aggregate(area_count="count()", groupby=["AREA NAME"])
        .transform_window(rank="rank(area_count)", sort=[alt.SortField("area_count", order="descending")])
        .transform_filter(alt.datum.rank <= 10)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("area_count:Q", title="Incident Count"),
            y=alt.Y("AREA NAME:N", sort="-x", title="Area"),
            tooltip=[
                alt.Tooltip("AREA NAME:N", title="Area"),
                alt.Tooltip("area_count:Q", title="Count"),
            ],
        )
    )
    if conditional:
        chart = chart.encode(color=alt.condition(area_pick, alt.value(AREA_GOLD), alt.value(BASE_GRAY), empty=False))
    else:
        chart = chart.encode(color=alt.value(color_value))
    return chart

area_chart = alt.layer(
    area_layer(
        alt.Chart(crime_top)
        .add_params(area_pick)
        .transform_filter("!generalised_area")
        .transform_filter(no_incident_selected)
        .transform_filter(no_brush_selected),
        conditional=True,
    ),
    area_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(brush_selected)
        .transform_filter(brush),
        color_value=BRUSH_BLUE,
    ),
    area_layer(
        alt.Chart(map_df)
        .transform_filter("!generalised_area")
        .transform_filter(incident_selected)
        .transform_filter(incident_pick),
        color_value=INCIDENT_BLUE,
    ),
    area_layer(
        alt.Chart(crime_top)
        .transform_filter("generalised_area")
        .transform_filter(area_selected)
        .transform_filter(area_pick),
        color_value=AREA_GOLD,
    ),
).properties(width=384, height=147, title="Area Summary")

# -----------------------------
# Dashboard layout: similar to requested HTML design
# -----------------------------
dashboard = (
    alt.vconcat(
        alt.hconcat(point_map, time_chart, spacing=15),
        alt.hconcat(area_chart, victim_chart, spacing=15),
        alt.hconcat(victim_age_chart, crime_category_chart, spacing=15),
        spacing=15,
    )
    .add_params(generalised_area)
    .resolve_scale(color="independent")
    .configure(background=PANEL_BG)
    .configure_view(stroke=LINE, continuousWidth=270, continuousHeight=200)
    .configure_axis(labelFontSize=10, titleFontSize=11, gridColor="#e8e1d8")
    .configure_title(fontSize=13, anchor="start", color="#202020")
    .properties(padding=8)
)

spec = dashboard.to_dict()
spec_json = json.dumps(spec)

# ── DR_NO → AREA NAME lookup (Python-injected, powers the generalise button) ──
# This encodes the two-level hierarchy: Incident (leaf) → Area (parent)
dr_to_area_map  = (
    map_df[["DR_NO", "AREA NAME"]]
    .dropna()
    .drop_duplicates(subset=["DR_NO"])
    .set_index("DR_NO")["AREA NAME"]
    .to_dict()
)
dr_to_area_json = json.dumps({str(k): v for k, v in dr_to_area_map.items()})

html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>System B Altair Dashboard</title>
  <script src=\"https://cdn.jsdelivr.net/npm/vega@5\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/vega-lite@5\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/vega-embed@6\"></script>
  <style>
    :root {{
      --bg: {PAGE_BG};
      --ink: #202020; --muted: #5a5a5a;
      --panel: {PANEL_BG}; --line: {LINE};
      --accent: {ACCENT};
      --gold: {AREA_GOLD}; --gold-soft:#fff8e0; --gold-border:#d4a800;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Georgia,"Times New Roman",serif;
            background:var(--bg); color:var(--ink); }}
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

    .controls {{
      display:flex; align-items:center; gap:10px;
      margin-bottom:14px; flex-wrap:wrap;
      background:var(--panel); border:1px solid var(--line);
      border-radius:4px; padding:10px 14px;
    }}
    .controls-label {{ font-size:0.85rem; color:var(--muted); }}
    .gen-status {{
      font-size:0.83rem; color:#7a5000;
      padding:3px 10px; border-radius:3px;
      border:1px solid var(--gold-border);
      background:var(--gold-soft); display:none;
    }}
    .gen-status.visible {{ display:inline-block; }}
    .spacer {{ flex:1; }}
    button {{
      padding:8px 18px; border:1px solid #cdbfaa;
      background:#fff; font-family:inherit; font-size:0.9rem;
      cursor:pointer; border-radius:3px; transition:all 0.15s;
    }}
    button:hover:not(:disabled) {{ background:#f5efe6; }}
    .btn-generalise {{
      border:2px solid var(--gold-border); color:#7a5000;
      background:#fff; font-weight:bold;
    }}
    .btn-generalise:hover:not(:disabled) {{ background:var(--gold-soft); }}
    .btn-generalise.active {{ background:var(--gold); color:#fff; border-color:var(--gold); }}
    .btn-generalise:disabled {{ opacity:0.35; cursor:not-allowed; }}
    .btn-primary {{ background:{ACCENT}; color:#fff; border-color:{ACCENT}; }}
    .btn-primary:hover {{ opacity:0.88; }}

    #vis.panel > div {{ width:100%; }}
    .note {{ margin-top:8px; color:var(--muted); font-size:0.82rem; text-align:center; }}
  </style>
</head>
<body>
<div class=\"wrap\">

    <section class=\"hero\">
        <h2>System B: Crime Data Analysis</h2>
    </section>

    <section class=\"hero\">
        <h3>Instructions: </h3>
        <ul style="margin-top:10px; line-height:1.6;">
        <li>Each point on the map reperesents a crime incident.</li>
        <li>Click a point on the map to select a crime incident.</li>
        <li>Click <b>Generalise</b> to expand from one incident to the whole area.</li>
        <li>Drag on the map to brush and compare multiple incidents.</li>
        <li>Use the charts to explore crime type, victim age, victim sex, and time patterns.</li>
        <li>The temporal heatmap shows crime counts by month and year</li>
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
  <!-- Controls: Generalise (Incident→Area) + status + Reset -->
  <div class=\"controls\">
    <span class=\"controls-label\">Generalised selection &nbsp;<b>Incident → Area</b>:</span>
    <button class=\"btn-generalise\" id=\"generaliseBtn\" disabled> Generalise </button>
    <span class=\"gen-status\" id=\"genStatus\"></span>
    <span class=\"spacer\"></span>
    <button class=\"btn-primary\" id=\"resetBtn\"> Reset </button>
  </div>

  <section id=\"vis\" class=\"panel\"></section>

</div>
<script>
  const spec     = {spec_json};
  // DR_NO → AREA NAME lookup — built from Python data at build time
  // Encodes the two-level hierarchy: Incident (leaf) → Area (parent)
  const drToArea = {dr_to_area_json};

  let viewRef      = null;
  let selectedDRNO = null;   // specific incident (leaf level)
  let selectedArea = null;   // its parent area   (general level)
  let genActive    = false;

  vegaEmbed('#vis', spec, {{ actions:false, renderer:'canvas' }}).then(res => {{
    viewRef = res.view;

    // Listen for clicks on map points (incident_pick signal)
    viewRef.addSignalListener('incident_pick', (_, val) => {{
      if (val && val.DR_NO && val.DR_NO.length) {{
        selectedDRNO = String(val.DR_NO[0]);
        selectedArea = drToArea[selectedDRNO] || null;
      }} else {{
        selectedDRNO = null;
        selectedArea = null;
        if (genActive) _deactivate();
      }}
      _updateUI();
    }});
  }});

  async function _activate() {{
    if (!viewRef || !selectedArea) return;
    genActive = true;

    // 1) Flip the boolean param → charts switch to area_pick layers
    await viewRef.signal('generalised_area', true).runAsync();

    // 2) Inject the parent area name into the area_pick selection store
    //    so all area_pick transform_filter calls match correctly
    viewRef.data('area_pick_store', [{{
      unit: "",
      fields: [{{ type: "E", field: "AREA NAME" }}],
      values: [[ selectedArea ]]
    }}]);
    await viewRef.runAsync();

    _updateUI();
  }}

  async function _deactivate() {{
    if (!viewRef) return;
    genActive = false;
    await viewRef.signal('generalised_area', false).runAsync();
    viewRef.data('area_pick_store', []);  // clear area selection
    await viewRef.runAsync();
    _updateUI();
  }}

  function _updateUI() {{
    const btn  = document.getElementById('generaliseBtn');
    const stat = document.getElementById('genStatus');

    if (genActive && selectedArea) {{
      btn.textContent = '🔽 Specific ↓';
      btn.classList.add('active'); btn.disabled = false;
      stat.innerHTML = `Area: <strong>${{selectedArea}}</strong>`;
      stat.classList.add('visible');
    }} else if (selectedDRNO && !genActive) {{
      btn.textContent = '🔼 Generalise ↑';
      btn.classList.remove('active'); btn.disabled = false;
      stat.innerHTML = `Incident <strong>${{selectedDRNO}}</strong> → <strong>${{selectedArea || '?'}}</strong>`;
      stat.classList.add('visible');
    }} else {{
      btn.textContent = '🔼 Generalise ↑';
      btn.classList.remove('active'); btn.disabled = true;
      stat.classList.remove('visible'); stat.textContent = '';
    }}
  }}

  document.getElementById('generaliseBtn').addEventListener('click', async () => {{
    if (!viewRef) return;
    if (genActive) await _deactivate();
    else           await _activate();
  }});

  document.getElementById('resetBtn').addEventListener('click', () => {{
    window.location.reload();
  }});
</script>
</body>
</html>
"""
Path(output_file).write_text(html, encoding="utf-8")
print(f"Saved dashboard to: {output_file}")
