# Crime Data Analysis & Visualisation Dashboard

An interactive, multi-system crime analytics dashboard built over **1 million+ Los Angeles crime records** (2020–2025), developed as a 5-member group project for the Information Visualisation (COMPSCI5078) coursework at the University of Glasgow (2026).

📹 **[Watch the demo video](https://www.youtube.com/watch?v=O58jhg4eqmA)**

---

## Overview

The project explores crime patterns across Los Angeles through three interconnected analytical systems, each targeting a different level of analysis — from city-wide overviews to incident-level drill-downs. All systems are linked through a shared dataset and a unified navigation dashboard.

Users can explore crime by time, geography, type, and victim demographics through brushing, filtering, and a novel **generalised selection** mechanism that traverses a semantic hierarchy from specific incidents to broader categories.

---

## System Architecture

| System | Focus | Approach |
|---|---|---|
| **System A** | City-wide overview | Binned density map + brushing/highlighting across 5 linked charts |
| **System B** | Incident-level detail | Point map + area summary + generalised selection (incident → area) |
| **System C** | Analytical exploration | Control-driven filtering + generalised selection (crime description → crime group) |
| **Dashboard** | Navigation hub | Unified entry point linking all three systems |

---

## Features

- 🗺️ **Three coordinated analytical systems** — each with a distinct spatial, temporal, and categorical perspective
- 🔗 **Fully linked visualisations** — brushing or selecting in one chart filters all others in real time
- 🔼 **Generalised selection** — one-click traversal from a specific incident or crime type up to its parent category or area
- 📅 **Temporal analysis** — monthly line charts and year-by-month heatmaps
- 👥 **Victim demographics** — age distribution, sex breakdown, and age band × crime group heatmap
- 🎨 **Perceptually uniform colour** — Viridis palette throughout System A for colour-blind accessibility
- 📊 **Usability evaluated** — tested with 10 participants across 6 structured tasks; System C rated highest on ease of use (3.86/5)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Visualisation | Altair, Vega-Lite |
| Frontend | HTML, CSS, JavaScript |
| Backend / Processing | Python, Pandas, NumPy |
| Dashboard Navigation | Vanilla JS + custom CSS |
| Dataset | LA Crime Data 2020–2025 (Kaggle / LAPD Open Data) |

---

## My Contributions

### System C — Full Design & Implementation
System C is a control-driven analytical dashboard focused on crime-type exploration. I designed and built it end-to-end:

- **Crime group filter** — dropdown allowing users to filter all charts by high-level crime group (Violent Crime, Property Crime, Domestic Violence, etc.)
- **Crime type ranking chart** — horizontal bar chart showing the most frequent detailed crime descriptions within the selected group
- **Temporal overview** — area line chart showing monthly crime trends, responsive to all active filters
- **Spatial distribution map** — point-based map with drag-to-brush region selection
- **Age band × crime group heatmap** — explicitly encodes how crime types vary across victim age bands (0–10, 10–20, ... 60+) in a single view
- **Victim sex composition chart** — stacked percentage bar chart showing female/male/unknown breakdown per crime group
- **Full multi-view linking** — all charts are connected; any selection propagates across every view simultaneously
- **Year range slider** — allows temporal subsetting across the entire dashboard
- **Area filter dropdown** — narrows all views to a specific LAPD reporting district

### Generalised Selection — Systems B & C
Generalised selection lets users move from a specific data point up one level in a semantic hierarchy in a single click:

- **System C** — clicking a crime description in the ranking chart and pressing "Generalise selection" switches all linked views from filtering by that specific description to filtering by its parent **Crime Group** (mapped via a `DESCTOGROUP` dictionary)
- **System B** — contributed to the equivalent spatial generalisation: selecting an incident (DRNO) and generalising jumps all views to filter by the incident's parent **AREA NAME** (resolved via a pre-calculated `drToArea` mapping)
- In both cases, generalisation preserves the user's context — it does not reset the dashboard, just moves one level up the hierarchy

Generalised selection allows a user to move **up a semantic hierarchy** — from a specific, hand-picked item to all items in its parent category — with a single click, without losing the context of the charts around them.

Both System B and System C implement this feature, but they apply it to different hierarchies:

| System | Leaf (specific) | Parent (general) | Hierarchy type |
|--------|----------------|------------------|----------------|
| **B**  | A single crime incident (`DR_NO`) | The LAPD reporting area that incident belongs to (`AREA NAME`) | Spatial |
| **C**  | A specific crime description (e.g. `BATTERY - SIMPLE ASSAULT`) | The crime group it belongs to (e.g. `Violent Crime`) | Conceptual |

### Frontend Dashboard
Built the complete navigation dashboard (`dashboard.html` + `dashboard.css`) that serves as the entry point for the entire project:

- Responsive navigation header linking all three systems
- System cards with descriptions, feature lists, and direct launch buttons
- Dataset info panel with source links
- Clean glassmorphism UI with Inter font, CSS variables, and mobile-responsive grid layout

---

## Project Structure

```
crime-dashboard/
├── dashboard.html          # ← Main navigation hub (built by me)
├── dashboard.css           # ← Full frontend styling (built by me)
├── index.html              # System A wrapper page
├── SYSTEM_A.html           # System A visualisation output
├── SYSTEM_A.py             # System A source (Altair)
├── SYSTEM_B.html           # System B visualisation output
├── SYSTEM_B.py             # System B source (Altair + JS)
├── SYSTEM_C.html           # System C visualisation output
├── SYSTEM_C.py             # System C source (built by me)
├── Crime_Data_from_2020_to_Present.csv   # Dataset (not included)
└── README.md
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/vaishnavibk29/crime-dashboard.git
cd crime-dashboard
```

### 2. Install dependencies
```bash
pip install pandas numpy altair
```

### 3. Get the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/ishajangir/crime-data) and place `Crime_Data_from_2020_to_Present.csv` in the root folder.

### 4. Generate the system HTML files
```bash
python SYSTEM_A.py    # outputs SYSTEM_A.html
python SYSTEM_B.py    # outputs SYSTEM_B.html
python SYSTEM_C.py    # outputs SYSTEM_C.html
```

### 5. Open the dashboard
Open `dashboard.html` in any modern browser — no server needed.

---

## Dataset

**LA Crime Data from 2020 to Present**
- Source: [Kaggle](https://www.kaggle.com/datasets/ishajangir/crime-data) / [LAPD Open Data](https://data.lacity.org)
- ~1 million records, 28 attributes
- Key fields used: date reported, area name, crime description, victim age, victim sex, latitude, longitude

**Preprocessing applied:**
- Removed irrelevant columns (Cross Street, Mocodes)
- Filtered victim ages to valid range (1–119), with exceptions for child-related crimes
- Standardised victim sex labels (M→Male, F→Female, invalid→Unknown)
- Classified ~200+ crime descriptions into 8 groups via keyword matching

---

## Evaluation Results

Evaluated with **10 participants** across 6 structured tasks per system using a within-subjects design.

| Metric | System A | System B | System C |
|---|---|---|---|
| Usability avg (out of 5) | 3.60 | 3.52 | **3.86** |
| Filtering intuitiveness | 3.60 | 3.40 | **4.00** |
| Ease of finding info | 3.70 | 3.50 | **3.90** |
| Best map votes | 4/10 | 4/10 | 2/10 |
| Easiest to use votes | 1/10 | 4/10 | **5/10** |
| Most preferred (ranked) | 1/10 | 4/10 | **5/10** |

System C achieved the highest usability score and was chosen as the most preferred system by 50% of participants, largely attributed to its structured filtering controls and the age band × crime group heatmap making relationships immediately readable.

---

## Design Decisions

**Why generalised selection?** Rather than re-filtering from the top down, generalisation always starts from a user-selected item and advances exactly one level up the hierarchy — mirroring how people naturally think when examining data (specific → broad).

**Why a control-driven layout in System C?** Placing filters prominently at the top (year range, area, crime group) lets users explicitly define the data subset before examining any chart, reducing ambiguity about what is currently shown.

**Why an age band × crime group heatmap?** Encoding two attributes in a single view removes the need to mentally combine separate charts — users immediately see how crime type distribution shifts across age groups, which no other system provided.

---

*built to make a million crimes legible.*
