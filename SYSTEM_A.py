#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import warnings
import json

#Altair imports
import altair as alt
alt.data_transformers.disable_max_rows()

#additional
from vega_datasets import data


#load data
file_path = "Crime_Data_from_2020_to_Present.csv"

df_crime = pd.read_csv(file_path)

#df_crime.head()


#checking data/exploration
#df_crime.dtypes

df_crime['Vict Age'].describe()

missing_values = df_crime.isnull().sum()
missing_percentage = round((missing_values / len(df_crime)) * 100, 2)
missing_percentage

#Drop Cross Street and Mocodes as they are not relevant in the analysis
df_crime.drop(['Cross Street' , 'Mocodes'] , axis = 1 ,inplace = True)

#Fill null values in Crime Codes with 'No Other Crime' as no other crimes present.
df_crime['Crm Cd 1'] = df_crime['Crm Cd 1'].fillna('No Other Crime')
df_crime['Crm Cd 2'] = df_crime['Crm Cd 2'].fillna('No Other Crime')
df_crime['Crm Cd 3'] = df_crime['Crm Cd 3'].fillna('No Other Crime')
df_crime['Crm Cd 4'] = df_crime['Crm Cd 4'].fillna('No Other Crime')

#Fill null values in Weapon Used Cd and Weapon Desc with relevant values
df_crime['Weapon Used Cd'] = df_crime['Weapon Used Cd'].fillna(0)
df_crime['Weapon Desc'] = df_crime['Weapon Desc'].fillna('No Weapon')

#Filtered the dataset to retain only records where victim age is between 0 and 119, removing invalid values outside this range.
df_crime = df_crime[(df_crime['Vict Age'] >= 0) & (df_crime['Vict Age'] < 120)]

# List of crime descriptions to keep
keep_crimes = ['CHILD NEGLECT (SEE 300 W.I.C.)',
               'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT',
               'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
               'KIDNAPPING',
               'KIDNAPPING - GRAND ATTEMPT',
               'CHILD STEALING',
               'CHILD ABANDONMENT']

# Filter the DataFrame
df_crime = df_crime[
    (df_crime['Vict Age'] != 0) |                                              # Keep rows where age is not 0
    (df_crime['Vict Age'] == 0) & (df_crime['Crm Cd Desc'].isin(keep_crimes))  # Keep rows where age is 0 and crime description is in the list
]

#Replace irrelevant data with 'NaN' and then replace with 'Unknown'
df_crime['Vict Sex'] = df_crime['Vict Sex'].replace({'X' : np.nan, 'H' : np.nan, '-' : np.nan})
df_crime['Vict Sex'] = df_crime['Vict Sex'].fillna('Unknown')

#Drop rows with NA values as there are low number of null values in irrelevant columns
df_crime.dropna(inplace=True)

# Convert object columns to string to avoid Altair/Vega export type errors
for col in df_crime.select_dtypes(include=['object']).columns:
    df_crime[col] = df_crime[col].fillna('').astype(str)

if len(df_crime) > 10000:
    df_crime = df_crime.sample(n=10000, random_state=42).copy()


#Reset index due to dropping rows
df_crime = df_crime.reset_index(drop = True)

#df_crime.describe()

#binning for categories, according to keyword, this is to improve filtering and categorisation
def classify_crime(crime_description):

    crime_description = crime_description.lower()
    # Sexual offenses
    if "rape" in crime_description or "sexual" in crime_description:
        return "Sexual Offenses"

    # Fraud / financial crime
    if "identity" in crime_description or "bunco" in crime_description:
        return "Fraud / Financial Crime"

    # Domestic violence
    if "intimate partner" in crime_description or "restraining order" in crime_description:
        return "Domestic Violence"

    # Threats / harassment
    if "threat" in crime_description or "lewd" in crime_description:
        return "Threats / Harassment"

    # Vehicle crimes
    if "vehicle" in crime_description:
        return "Vehicle Crime"

    # Vandalism / trespass
    if "vandalism" in crime_description or "trespass" in crime_description:
        return "Vandalism / Trespass"

    # Violent crimes
    if any(word in crime_description for word in [
        "assault", "battery", "robbery", "brandish"
    ]):
        return "Violent Crime"

    # Property crimes
    if any(word in crime_description for word in [
        "burglary", "theft", "shoplifting", "pickpocket", "bike"
    ]):
        return "Property Crime"

    return "Other"


# In[19]:


import altair as alt
import pandas as pd

# =========================================================
# Step 1: data preparation
# =========================================================
visual_df = df_crime.copy()

visual_df["Date Rptd"] = pd.to_datetime(
    visual_df["Date Rptd"],
    errors="coerce"
)

visual_df = visual_df.dropna(
    subset=["Date Rptd", "Crm Cd Desc", "Vict Age", "Vict Sex"]
).copy()

visual_df["crime_category"] = visual_df["Crm Cd Desc"].apply(classify_crime)
visual_df["year_month"] = visual_df["Date Rptd"].dt.to_period("M").dt.to_timestamp()
visual_df["year"] = visual_df["Date Rptd"].dt.year.astype(str)

# Remove incomplete 2025 data
visual_df = visual_df[
    visual_df["year_month"].dt.year < 2025
].copy()

# Keep sensible ages only
visual_df = visual_df[
    (visual_df["Vict Age"] > 0) &
    (visual_df["Vict Age"] < 100)
].copy()

# Clean victim sex labels
visual_df["Vict Sex"] = visual_df["Vict Sex"].replace({
    "M": "Male",
    "F": "Female",
    "X": "Unknown"
})

visual_df["Vict Sex"] = visual_df["Vict Sex"].where(
    visual_df["Vict Sex"].isin(["Male", "Female", "Unknown"]),
    "Unknown"
)

# Pre-aggregate monthly counts for the temporal chart
temporal_df = (
    visual_df
    .groupby("year_month", as_index=False)
    .size()
    .rename(columns={"size": "monthly_crime_count"})
    .sort_values("year_month")
)

# =========================================================
# Step 2: fixed axis limits
# =========================================================
sex_count_summary = (
    visual_df.groupby("Vict Sex")
    .size()
    .reindex(["Male", "Female", "Unknown"], fill_value=0)
)
max_sex_count = int(sex_count_summary.max())

age_bin_edges = list(range(0, 105, 5))
age_bin_counts = pd.cut(
    visual_df["Vict Age"],
    bins=age_bin_edges,
    right=False,
    include_lowest=True
).value_counts().sort_index()
max_age_bin_count = int(age_bin_counts.max())

crime_category_counts = visual_df.groupby("crime_category").size()
max_crime_count = int(crime_category_counts.max())

monthly_counts = visual_df.groupby("year_month").size()
max_monthly_count = int(monthly_counts.max())

# =========================================================
# Step 3: colours
# =========================================================
viridis_yellow = "#fde725"
viridis_green = "#5ec962"
viridis_teal = "#21918c"
viridis_blue = "#3b528b"
viridis_purple = "#440154"
soft_lilac = "#ddd7ef"
muted_sex_unselected = "#eceaf4"
light_blue_fill = "#9ecae1"

# =========================================================
# Step 4: interactive selections
# Double-click clears selections
# =========================================================
crime_selection = alt.selection_point(
    name="crime_selection",
    fields=["crime_category"],
    empty="all",
    clear="dblclick"
)

time_brush = alt.selection_interval(
    name="time_brush",
    encodings=["x"],
    clear="dblclick"
)

sex_selection = alt.selection_point(
    name="sex_selection",
    fields=["Vict Sex"],
    empty="all",
    clear="dblclick"
)

age_brush = alt.selection_interval(
    name="age_brush",
    encodings=["x"],
    clear="dblclick"
)

map_brush = alt.selection_interval(
    name="map_brush",
    encodings=["x", "y"],
    clear="dblclick"
)


# =========================================================
# Step 6: map data and stable colour scale
# =========================================================
map_df = visual_df[
    visual_df["LAT"].notna() &
    visual_df["LON"].notna() &
    (visual_df["LAT"] != 0) &
    (visual_df["LON"] != 0) &
    visual_df["LAT"].between(33.70, 34.35) &
    visual_df["LON"].between(-118.70, -118.15)
].copy()

map_bin_dataframe = map_df.copy()

map_bin_dataframe["longitude_bin"] = pd.cut(
    map_bin_dataframe["LON"],
    bins=70
)

map_bin_dataframe["latitude_bin"] = pd.cut(
    map_bin_dataframe["LAT"],
    bins=70
)

max_map_bin_count = int(
    map_bin_dataframe
    .groupby(["longitude_bin", "latitude_bin"])
    .size()
    .max()
)

# =========================================================
# Step 7: spatial overview map
# =========================================================
spatial_map = (
    alt.Chart(map_df)
    .mark_rect()
    .encode(
        x=alt.X(
            "LON:Q",
            bin=alt.Bin(maxbins=70),
            title="Longitude",
            scale=alt.Scale(domain=[-118.78, -118.05]),
            axis=alt.Axis(grid=False)
        ),
        y=alt.Y(
            "LAT:Q",
            bin=alt.Bin(maxbins=70),
            title="Latitude",
            scale=alt.Scale(domain=[33.65, 34.42]),
            axis=alt.Axis(grid=False)
        ),
        color=alt.Color(
            "count():Q",
            title="Incident Count",
            scale=alt.Scale(
                domain=[0, max_map_bin_count],
                range=[
                    viridis_yellow,
                    viridis_green,
                    viridis_teal,
                    viridis_blue,
                    viridis_purple
                ],
                nice=False
            )
        ),
        tooltip=[
            alt.Tooltip("count():Q", title="Incidents")
        ]
    )
    .add_params(map_brush)
    .transform_filter(crime_selection)
    .transform_filter(time_brush)
    .transform_filter(sex_selection)
    .transform_filter(age_brush)
    .properties(
        width=650,
        height=300,
        title="Spatial Overview of Crime"
    )
)

# =========================================================
# Step 8: crime category chart
# =========================================================
crime_base = (
    alt.Chart(visual_df)
    .mark_bar(color=soft_lilac)
    .encode(
        y=alt.Y(
            "crime_category:N",
            sort="-x",
            title="Crime Category"
        ),
        x=alt.X(
            "count()",
            title="Number of Incidents",
            scale=alt.Scale(domain=[0, max_crime_count], nice=False)
        )
    )
)

crime_highlight = (
    alt.Chart(visual_df)
    .mark_bar()
    .encode(
        y=alt.Y("crime_category:N", sort="-x"),
        x=alt.X("count()"),
        color=alt.condition(
            crime_selection,
            alt.value(viridis_purple),
            alt.value("transparent")
        )
    )
    .transform_filter(age_brush)
    .transform_filter(map_brush)
    .transform_filter(time_brush)
    .transform_filter(sex_selection)
)

crime_chart = (
    (crime_base + crime_highlight)
    .add_params(crime_selection)
    .properties(
        width=310,
        height=210,
        title="Crime Categories"
    )
)

# =========================================================
# Step 9: temporal chart (interactive, but not filtered by map)
# =========================================================
time_chart = (
    alt.Chart(visual_df)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "year_month:T",
            title="Date"
        ),
        y=alt.Y(
            "count()",
            title="Reported Crimes",
            scale=alt.Scale(domain=[0, max_monthly_count], nice=False)
        ),
        tooltip=[
            alt.Tooltip("year_month:T", title="Month"),
            alt.Tooltip("count()", title="Number of Incidents")
        ]
    )
    .add_params(time_brush)
    .transform_filter(crime_selection)
    .transform_filter(sex_selection)
    .transform_filter(age_brush)
    .transform_filter(map_brush)
    .properties(
        width=350,
        height=200,
        title="Temporal Overview of Crime"
    )
)
# =========================================================
# Step 10: victim age chart
# =========================================================
age_chart = (
    alt.Chart(visual_df)
    .mark_bar(
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2
    )
    .encode(
        x=alt.X(
            "Vict Age:Q",
            bin=alt.Bin(step=5),
            title="Victim Age",
            scale=alt.Scale(domain=[0, 100], nice=False)
        ),
        y=alt.Y(
            "count()",
            title="Number of Victims",
            scale=alt.Scale(domain=[0, max_age_bin_count], nice=False)
        ),
        color=alt.value(viridis_teal),
        tooltip=[
            alt.Tooltip("count()", title="Victims")
        ]
    )
    .add_params(age_brush)
    .transform_filter(map_brush)
    .transform_filter(crime_selection)
    .transform_filter(time_brush)
    .transform_filter(sex_selection)
    .properties(
        width=350,
        height=200,
        title="Victim Age Distribution"
    )
)

# =========================================================
# Step 11: victim sex chart
# =========================================================
sex_chart = (
    alt.Chart(visual_df)
    .mark_bar(cornerRadiusEnd=3)
    .encode(
        y=alt.Y(
            "Vict Sex:N",
            title="Victim Sex",
            sort=["Male", "Female", "Unknown"]
        ),
        x=alt.X(
            "count()",
            title="Number of Victims",
            scale=alt.Scale(domain=[0, max_sex_count], nice=False)
        ),
        color=alt.condition(
            sex_selection,
            alt.Color(
                "Vict Sex:N",
                scale=alt.Scale(
                    domain=["Male", "Female", "Unknown"],
                    range=[viridis_blue, viridis_yellow, viridis_teal]
                ),
                legend=None
            ),
            alt.value(muted_sex_unselected)
        ),
        tooltip=[
            alt.Tooltip("Vict Sex:N", title="Sex"),
            alt.Tooltip("count()", title="Victims")
        ]
    )
    .add_params(sex_selection)
    .transform_filter(map_brush)
    .transform_filter(crime_selection)
    .transform_filter(time_brush)
    .transform_filter(age_brush)
    .properties(
        width=320,
        height=200,
        title="Victim Sex Distribution"
    )
)

# =========================================================
# Step 12: dashboard layout
# =========================================================
dashboard = (
    alt.vconcat(
        spatial_map,
        crime_chart | time_chart,
        sex_chart | age_chart,
        spacing=18
    )
    .resolve_scale(y="independent")
    .configure_view(
        stroke="#d8d3cc"
    )
    .configure_axis(
        labelFontSize=11,
        titleFontSize=12,
        gridColor="#e8e3dc"
    )
    .configure_title(
        fontSize=20,
        anchor="middle",
        color=viridis_purple
    )
    .configure_legend(
        labelFontSize=11,
        titleFontSize=12
    )
    .properties(
        title="Interactive Exploration of Crime Patterns"
    )
)

dashboard
dashboard.save("SYSTEM_A.html")






