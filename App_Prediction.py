"""
ROAD SAFETY INTELLIGENCE
UK STATS19 collision analytics and research-only severity prediction.

This application presents the work in STATS19_Capstone_Rich_CRISPDM_V6.ipynb:

* descriptive dashboards over EVERY year in the STATS19 extract (the supplied
  five-year files cover 2021-2025: 513,801 collisions, 937,265 vehicle records
  and 652,821 casualty records);
* the notebook's statistical evidence (hypotheses H1-H7), read from outputs/;
* the selected weighted neural-network pipeline saved in models/, driven by
  the notebook's input contract, together with its evaluation results.

Run:
    streamlit run App_Prediction.py

Expected project layout:
    App_Prediction.py
    requirements.txt
    data/      collision, casualty and vehicle CSVs - either the DfT
               "last 5 years" files or several annual files; each table is
               recognised from its columns, and all years found are combined
    models/    stats19_severity_model.joblib, model_contract.json, metadata.json
    outputs/   evidence tables written by the notebook

The model predicts severity conditional on a recorded collision. It does not
predict whether a collision will occur, and it did not pass the notebook's
acceptance gate, so it is a research demonstration only.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. PAGE CONFIGURATION, PATHS AND STREAMLIT COMPATIBILITY
# =============================================================================

st.set_page_config(
    page_title="Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
OUTPUT_DIR = APP_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "stats19_severity_model.joblib"
CONTRACT_PATH = MODEL_DIR / "model_contract.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
NOTEBOOK_NAME = "STATS19_Capstone_Rich_CRISPDM_V6.ipynb"

try:
    ST_VERSION = tuple(int(part) for part in st.__version__.split(".")[:2])
except ValueError:
    ST_VERSION = (1, 50)

_STRETCH_KWARGS: dict[Any, dict] = {}


def stretch(widget) -> dict:
    """Full-width keyword for this Streamlit version.

    Streamlit 1.50+ uses width="stretch"; older releases use
    use_container_width=True, which newer releases deprecate.
    """
    if widget not in _STRETCH_KWARGS:
        try:
            params = inspect.signature(widget).parameters
        except (TypeError, ValueError):
            params = {}
        if ST_VERSION >= (1, 50) and "width" in params:
            _STRETCH_KWARGS[widget] = {"width": "stretch"}
        elif "use_container_width" in params:
            _STRETCH_KWARGS[widget] = {"use_container_width": True}
        else:
            _STRETCH_KWARGS[widget] = {}
    return _STRETCH_KWARGS[widget]


# =============================================================================
# 2. STATS19 CODE LABELS
# Labels follow the notebook's STATS19_LABELS (Section 3.8) and extend them to
# the other coded fields used by the dashboard and the model. Codes without a
# verified label are shown as "STATS19 code N" rather than guessed.
# =============================================================================

MISSING = "Data missing or out of range"
SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLOURS = {"Fatal": "#FF5C70", "Serious": "#FFB547", "Slight": "#35C6A5"}

STATS19_LABELS: dict[str, dict[int, str]] = {
    "collision_severity": {1: "Fatal", 2: "Serious", 3: "Slight"},
    "road_type": {
        1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
        6: "Single carriageway", 7: "Slip road", 9: "Unknown road type", -1: MISSING,
    },
    "junction_detail": {
        # Earlier STATS19 coding
        0: "Not at or within 20 metres of junction", 1: "Roundabout",
        2: "Mini-roundabout", 3: "T or staggered junction", 5: "Slip road",
        6: "Crossroads", 7: "More than four arms",
        8: "Private drive or entrance", 9: "Other junction",
        # Revised coding present in the five-year extract
        13: "T or staggered junction", 16: "Crossroads",
        17: "More than four arms", 18: "Private drive or entrance",
        19: "Other junction", 99: "Unknown junction detail", -1: MISSING,
    },
    "junction_control": {
        0: "Not at junction or within 20 metres", 1: "Authorised person",
        2: "Auto traffic signal", 3: "Stop sign", 4: "Give way or uncontrolled",
        9: "Unknown junction control", -1: MISSING,
    },
    "light_conditions": {
        1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
        6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: MISSING,
    },
    "weather_conditions": {
        1: "Fine - no high winds", 2: "Raining - no high winds",
        3: "Snowing - no high winds", 4: "Fine - high winds",
        5: "Raining - high winds", 6: "Snowing - high winds",
        7: "Fog or mist", 8: "Other weather", 9: "Unknown weather", -1: MISSING,
    },
    "road_surface_conditions": {
        1: "Dry", 2: "Wet or damp", 3: "Snow", 4: "Frost or ice",
        5: "Flood over 3 cm deep", 9: "Unknown surface condition", -1: MISSING,
    },
    "special_conditions_at_site": {
        0: "None", 1: "Auto traffic signal - out", 2: "Auto signal part defective",
        3: "Road sign or marking defective or obscured", 4: "Roadworks",
        5: "Road surface defective", 6: "Oil or diesel", 7: "Mud",
        9: "Unknown special conditions", -1: MISSING,
    },
    # Revised hazard codes 11-21 are left as "STATS19 code N": see the DfT
    # data guide for their definitions.
    "carriageway_hazards": {0: "None", 99: "Unknown carriageway hazard", -1: MISSING},
    "urban_or_rural_area": {1: "Urban", 2: "Rural", 3: "Unallocated", -1: MISSING},
    "trunk_road_flag": {1: "Trunk road", 2: "Non-trunk road", -1: MISSING},
    "first_road_class": {
        1: "Motorway", 2: "A(M)", 3: "A road", 4: "B road", 5: "C road",
        6: "Unclassified", -1: MISSING,
    },
    "second_road_class": {
        0: "Not at junction or within 20 metres", 1: "Motorway", 2: "A(M)",
        3: "A road", 4: "B road", 5: "C road", 6: "Unclassified",
        9: "Unknown second road class", -1: MISSING,
    },
    "day_of_week": {
        1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
        5: "Thursday", 6: "Friday", 7: "Saturday",
    },
    "month": {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
        6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
        11: "November", 12: "December",
    },
    "vehicle_type": {
        1: "Pedal cycle", 2: "Motorcycle 50cc and under",
        3: "Motorcycle 125cc and under", 4: "Motorcycle 125cc-500cc",
        5: "Motorcycle over 500cc", 8: "Taxi or private hire car", 9: "Car",
        10: "Minibus (8-16 passenger seats)", 11: "Bus or coach (17+ seats)",
        16: "Ridden horse", 17: "Agricultural vehicle", 18: "Tram",
        19: "Van or goods 3.5t and under", 20: "Goods over 3.5t and under 7.5t",
        21: "Goods 7.5t and over", 22: "Mobility scooter",
        23: "Electric motorcycle",
        33: "E-scooter",  # 98% of code-33 vehicles carry escooter_flag = 1
        90: "Other vehicle", 97: "Motorcycle - unknown cc",
        98: "Goods vehicle - unknown weight", 99: "Unknown vehicle type", -1: MISSING,
    },
    "vehicle_manoeuvre": {
        1: "Reversing", 2: "Parked", 3: "Waiting to go - held up",
        4: "Slowing or stopping", 5: "Moving off", 6: "U-turn",
        7: "Turning left", 8: "Waiting to turn left", 9: "Turning right",
        10: "Waiting to turn right", 11: "Changing lane to left",
        12: "Changing lane to right", 13: "Overtaking moving vehicle - offside",
        14: "Overtaking stationary vehicle - offside", 15: "Overtaking - nearside",
        16: "Going ahead - left-hand bend", 17: "Going ahead - right-hand bend",
        18: "Going ahead - other",
        19: "Going ahead",  # revised code replacing 16-18 in the five-year extract
        99: "Unknown manoeuvre", -1: MISSING,
    },
    "sex_of_driver": {1: "Male", 2: "Female", 3: "Not known", -1: MISSING},
    "sex_of_casualty": {1: "Male", 2: "Female", 9: "Unknown", -1: MISSING},
    "casualty_class": {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"},
    "age_band": {
        1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20", 5: "21–25", 6: "26–35",
        7: "36–45", 8: "46–55", 9: "56–65", 10: "66–75", 11: "Over 75",
        -1: "Unknown",
    },
}
FIELD_ALIASES = {
    "vehicle_type_first": "vehicle_type",
    "driver_sex_first": "sex_of_driver",
    "vehicle_manoeuvre_first": "vehicle_manoeuvre",
    "casualty_severity": "collision_severity",
    "age_band_of_casualty": "age_band",
    "age_band_of_driver": "age_band",
}
AGE_BAND_ORDER = list(STATS19_LABELS["age_band"].values())
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOUR_ORDER = [f"{h:02d}:00" for h in range(24)]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

USER_GROUP = {
    0: "Pedestrian", 1: "Cyclist", 8: "Car occupant", 9: "Car occupant",
    10: "Bus/coach occupant", 11: "Bus/coach occupant",
    19: "Van/goods occupant", 20: "Van/goods occupant",
    21: "Van/goods occupant", 98: "Van/goods occupant",
    33: "E-scooter rider",
}
USER_GROUP.update({code: "Motorcyclist" for code in (2, 3, 4, 5, 23, 97, 103, 104, 105, 106)})
VULNERABLE_USERS = ["Pedestrian", "Cyclist", "Motorcyclist"]

FEATURE_LABELS = {
    "road_type": "Road type",
    "speed_limit": "Speed limit (mph)",
    "junction_detail": "Junction detail",
    "junction_control": "Junction control",
    "light_conditions": "Light conditions",
    "weather_conditions": "Weather",
    "road_surface_conditions": "Road surface",
    "special_conditions_at_site": "Special conditions at site",
    "carriageway_hazards": "Carriageway hazards",
    "urban_or_rural_area": "Urban or rural area",
    "trunk_road_flag": "Trunk road",
    "first_road_class": "First road class",
    "second_road_class": "Second road class",
    "day_of_week": "Day of week",
    "month": "Month",
    "hour": "Hour of day",
    "time_band": "Time band",
    "weekend_flag": "Weekend",
    "number_of_vehicles": "Number of vehicles",
    "driver_age_mean": "Mean driver age",
    "driver_age_min": "Youngest driver age",
    "vehicle_age_mean": "Mean vehicle age (years)",
    "engine_cc_mean": "Mean engine capacity (cc)",
    "vehicle_type_first": "First vehicle type",
    "driver_sex_first": "First driver sex",
    "vehicle_manoeuvre_first": "First vehicle manoeuvre",
    "vehicle_records_count": "Vehicle records in collision",
    "calendar_year": "Calendar year",
    "casualty_type": "Casualty type",
    "age_band_of_casualty": "Casualty age band",
    "pedestrian_location": "Pedestrian location",
    "pedestrian_movement": "Pedestrian movement",
    "casualty_class": "Casualty class",
    "sex_of_casualty": "Casualty sex",
    "car_passenger": "Car passenger",
    "casualty_imd_decile": "Casualty IMD decile",
    "bus_or_coach_passenger": "Bus or coach passenger",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, str(name).replace("_", " ").capitalize())


def code_label(field: str, code) -> str:
    """Readable label for one STATS19 code, without inventing meanings."""
    mapping = STATS19_LABELS.get(FIELD_ALIASES.get(field, field))
    if mapping is None:
        return str(code)
    try:
        value = int(float(code))
    except (TypeError, ValueError):
        return str(code)
    return mapping.get(value, f"STATS19 code {value}")


def label_codes(series: pd.Series, field: str, unknown: str = "Unknown") -> pd.Series:
    """Vectorised code-to-label mapping returning a memory-light categorical."""
    mapping = STATS19_LABELS.get(FIELD_ALIASES.get(field, field), {})
    numeric = pd.to_numeric(series, errors="coerce")
    labels = numeric.map(mapping).astype(object)
    unmapped = labels.isna() & numeric.notna()
    if unmapped.any():
        labels.loc[unmapped] = "STATS19 code " + numeric[unmapped].astype(int).astype(str)
    return labels.fillna(unknown).astype("category")


# =============================================================================
# 3. GLOBAL VISUAL DESIGN
# =============================================================================

INK = "#EAF1FA"
MUTED = "#9AAAC0"
BG = "#10151D"
PANEL = "#1A222D"
PANEL_ALT = "#222D3A"
BORDER = "#354252"
BLUE = "#66A6FF"
TEAL = "#35C6A5"
AMBER = "#F4B860"
RED = "#F06C79"
PURPLE = "#B39DDB"
GREEN = "#77C98B"
CORAL = "#F28C68"
CYAN = "#63C7D8"
YEAR_PALETTE = [BLUE, TEAL, AMBER, PURPLE, RED, CYAN, CORAL, GREEN]
HYPOTHESIS_COLOURS = {
    "H1 Road and environment": BLUE,
    "H2 Driver and vehicle": AMBER,
    "H3 Temporal": PURPLE,
    "H4 Casualty": TEAL,
}

RESEARCH_ONLY_NOTICE = (
    "RESEARCH / DEMONSTRATION ONLY — this model has not passed the project's "
    "acceptance gate. Do not use its predictions for operational road-safety, "
    "emergency response, enforcement, insurance or individual decisions."
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {{
      color-scheme: dark;
      --bg: {BG}; --panel: {PANEL}; --panel2: {PANEL_ALT};
      --ink: {INK}; --muted: {MUTED}; --border: {BORDER};
      --blue: {BLUE}; --teal: {TEAL}; --amber: {AMBER}; --red: {RED};
    }}
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
      font-family: 'DM Sans', sans-serif !important;
      background: {BG} !important;
      color: {INK} !important;
    }}
    .stApp {{ background: {BG} !important; }}
    [data-testid="stHeader"] {{ background: rgba(8,19,33,.92) !important; }}
    [data-testid="stToolbar"] {{ right: 1rem; }}
    .block-container {{ max-width: 1600px; padding-top: 1.35rem; padding-bottom: 3rem; }}
    h1,h2,h3,h4,h5,p,li,label {{ color: {INK}; }}
    h1,h2,h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }}
    h1 {{ font-size: clamp(2rem, 3vw, 2.8rem) !important; font-weight: 700 !important; }}
    h2 {{ font-size: 1.55rem !important; }}
    h3 {{ font-size: 1.12rem !important; }}
    p, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(180deg,#151F2B 0%,#111923 100%) !important; border-right: 1px solid {BORDER}; }}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .72rem; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding: 8px 10px; border-radius: 9px; }}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background: rgba(99,199,216,.10); }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div {{ border:1px solid {BORDER} !important; border-radius:10px !important; }}
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {{ background:{PANEL_ALT} !important; }}
    [data-baseweb="popover"] [role="option"] {{ color:{INK} !important; }}
    [data-testid="stSidebar"] * {{ color: {INK} !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {{ background: {PANEL_ALT} !important; color: {INK} !important; }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color: {MUTED} !important; }}
    [data-testid="stMetric"] {{ min-height:118px; display:flex; flex-direction:column; justify-content:center;
      background: linear-gradient(145deg, {PANEL_ALT}, {PANEL}) !important;
      border: 1px solid {BORDER} !important; border-radius: 14px !important;
      padding: 16px 18px !important;
    }}
    [data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-size: .78rem !important; }}
    [data-testid="stMetricValue"] {{ color: {INK} !important; font-family: 'Space Grotesk',sans-serif; }}
    [data-testid="stMetricDelta"] {{ color: {TEAL} !important; }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
      background: {PANEL} !important; border: 1px solid {BORDER} !important;
      border-radius: 16px !important;
    }}
    [data-testid="stExpander"] {{ background: {PANEL} !important; border: 1px solid {BORDER} !important; border-radius: 12px !important; }}
    [data-testid="stAlert"] {{ border-radius: 12px !important; }}
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
      background: {PANEL} !important; border: 1px solid {BORDER}; border-radius: 12px;
    }}
    [data-baseweb="tab-list"] {{ gap: 8px; background: transparent; }}
    button[data-baseweb="tab"] {{ color: {MUTED} !important; background: {PANEL} !important; border-radius: 10px 10px 0 0; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEAL} !important; border-bottom: 2px solid {TEAL} !important; }}
    .section-kicker {{ color: {TEAL}; font-size: .68rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; margin-bottom: 4px; }}
    .section-title {{ color: {INK}; font-family: 'Space Grotesk',sans-serif; font-size: 1.55rem; font-weight: 700; margin: 0 0 4px; }}
    .coverage {{ color: {MUTED}; font-size: .86rem; margin: -0.9rem 0 1.4rem; line-height: 1.6; }}
    .coverage b {{ color: {INK}; font-weight: 700; }}
    .panel-heading {{ margin: 0 0 16px; width:100%; box-sizing:border-box; }}
    .panel-title {{ color: {INK}; font-family: 'Space Grotesk',sans-serif; font-size: 1rem; font-weight: 650; line-height: 1.35; margin: 0 0 7px; }}
    .panel-note {{ color: {MUTED}; font-size: .76rem; line-height: 1.5; margin: 0; }}
    .form-group-title {{ color: {TEAL}; font-size: .72rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin: 6px 0 2px; }}
    .notice {{
      background: rgba(255,181,71,.09); border: 1px solid rgba(255,181,71,.38);
      border-left: 4px solid {AMBER}; border-radius: 12px; color: #FFE0A6;
      padding: 13px 16px; font-size: .82rem; line-height: 1.55; margin: 12px 0 18px;
    }}
    .findings {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 14px 18px; margin: 0 0 1.6rem; }}
    .findings li {{ color: #C9D5E3; font-size: .86rem; line-height: 1.6; }}
    .findings .caution {{ color: {MUTED}; font-size: .74rem; margin-top: 6px; }}
    .result-card {{ background: linear-gradient(135deg,#263A3A,#202B37); border:1px solid {BORDER}; border-radius:16px; padding:20px 22px; }}
    .result-label {{ color:{MUTED}; font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }}
    .result-value {{ font-family:'Space Grotesk',sans-serif; font-size:2.2rem; font-weight:700; margin-top:5px; }}
    .small-muted {{ color:{MUTED}; font-size:.76rem; }}
    .filter-caption {{ color:{MUTED}; font-size:.72rem; line-height:1.45; }}
    .footer {{ border-top:1px solid {BORDER}; margin-top:38px; padding-top:16px; color:{MUTED}; font-size:.74rem; }}
    div.stButton > button[kind="primary"], button[kind="primary"] {{
      background: {TEAL} !important; color: #062018 !important; border: 0 !important; font-weight: 800 !important;
      border-radius: 10px !important;
    }}
    div.stButton > button {{
      border: 1px solid {BORDER} !important; border-radius: 10px !important;
      color: {INK} !important; background: {PANEL_ALT} !important;
    }}
    div[data-testid="stDownloadButton"] button {{
      background: linear-gradient(135deg, #14b8a6, #0f766e) !important; color: white !important;
      border: 1px solid rgba(20,184,166,.4) !important; border-radius: 10px !important;
      padding: .65rem 1.2rem; font-weight: 600; transition: all .25s ease-in-out;
    }}
    div[data-testid="stDownloadButton"] button:hover {{
      background: linear-gradient(135deg, #2dd4bf, #0d9488) !important; border-color: #5eead4 !important;
      transform: translateY(-2px); box-shadow: 0 6px 18px rgba(20,184,166,.25);
    }}
    [data-testid="stForm"] {{ border:1px solid {BORDER}; border-radius:14px; padding:20px; background:{PANEL} !important; }}
    [data-testid="stForm"] label {{ color:{INK} !important; font-weight:600 !important; }}
    [data-testid="stForm"] [data-baseweb="select"] > div,
    [data-testid="stForm"] input,
    [data-testid="stForm"] textarea {{ background:#0D141D !important; color:#F3F7FC !important; border:1px solid #64758A !important; border-radius:9px !important; }}
    [data-testid="stForm"] [data-baseweb="select"] svg {{ fill:#DCE7F4 !important; }}
    input, textarea {{ color: {INK} !important; }}
    /* Responsive layout: ordinary chart/form columns stack on phones.
       KPI columns have their own grid so the six cards remain compact. */
    [data-testid="stHorizontalBlock"] {{ align-items:stretch; gap:1.15rem; min-width:0; }}
    [data-testid="column"], [data-testid="stColumn"] {{ min-width:0 !important; box-sizing:border-box; }}
    [data-testid="stVegaLiteChart"], [data-testid="stAltairChart"] {{ width:100% !important; max-width:100%; overflow:hidden; }}
    [data-testid="stVegaLiteChart"] canvas, [data-testid="stVegaLiteChart"] svg {{ max-width:100%; }}
    [class*="st-key-kpi-grid"] {{ width:100%; min-width:0; margin:0 0 2.2rem !important; }}
    [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] {{
      display:grid !important; grid-template-columns:repeat(6,minmax(0,1fr)) !important;
      column-gap:1rem !important; row-gap:1.15rem !important;
      align-items:stretch !important; width:100% !important;
    }}
    [class*="st-key-kpi-grid"] [data-testid="column"],
    [class*="st-key-kpi-grid"] [data-testid="stColumn"] {{
      min-width:0 !important; width:auto !important; max-width:none !important;
      flex:none !important; margin:0 !important;
    }}
    .kpi-card {{ width:100%; min-width:0; min-height:112px; height:112px; box-sizing:border-box;
      display:flex; flex-direction:column; justify-content:center; overflow:hidden;
      padding:14px 13px; border-radius:13px; }}
    .kpi-label {{ font-size:.64rem; line-height:1.3; font-weight:800; letter-spacing:.065em; overflow-wrap:anywhere; }}
    .kpi-value {{ margin-top:7px; font-family:'Space Grotesk',sans-serif; font-size:clamp(1.12rem,1.5vw,1.48rem);
      line-height:1.1; font-weight:700; white-space:nowrap; letter-spacing:-.04em; font-variant-numeric:tabular-nums; }}
    @media(max-width: 1100px) {{
      .block-container {{ padding:1rem 1rem 2rem; }}
      [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] {{ grid-template-columns:repeat(3,minmax(0,1fr)) !important; }}
      .kpi-value {{ font-size:1.3rem; }}
    }}
    @media(max-width: 760px) {{
      .block-container {{ padding:1rem .9rem 2rem; }}
      [data-testid="stHorizontalBlock"] {{ flex-wrap:wrap !important; gap:1rem !important; }}
      [data-testid="stHorizontalBlock"] > [data-testid="column"],
      [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        flex:1 1 100% !important; width:100% !important; max-width:100% !important; min-width:100% !important;
      }}
      [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] {{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; column-gap:.9rem !important; row-gap:1.1rem !important; }}
      [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] > [data-testid="column"],
      [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
        min-width:0 !important; width:auto !important; max-width:none !important; flex:none !important;
      }}
      .kpi-card {{ min-height:104px; height:104px; padding:12px 10px; }}
      .kpi-label {{ font-size:.59rem; letter-spacing:.04em; }}
      .kpi-value {{ font-size:1.12rem; margin-top:6px; }}
      .panel-heading {{ margin-bottom:14px; }}
    }}
    @media(max-width: 480px) {{
      .block-container {{ padding:.75rem .65rem 1.5rem; }}
      .section-title {{ font-size:1.3rem; }}
      [class*="st-key-kpi-grid"] [data-testid="stHorizontalBlock"] {{ grid-template-columns:repeat(2,minmax(0,1fr)) !important; column-gap:.8rem !important; row-gap:1rem !important; }}
      .kpi-card {{ min-height:98px; height:98px; padding:11px 9px; }}
      .kpi-label {{ font-size:.57rem; }}
      .kpi-value {{ font-size:1.08rem; }}
      .panel-title {{ font-size:.96rem; }}
      .panel-note {{ font-size:.73rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 4. DATA LOADING - EVERY YEAR IN THE STATS19 EXTRACT
# The loaded frames are cached with st.cache_resource (shared, not copied on
# each rerun, which matters at ~2.1 million rows). Treat them as read-only.
# =============================================================================

COLLISION_COLUMNS = [
    "collision_index", "collision_year", "date", "time", "collision_severity",
    "urban_or_rural_area", "speed_limit", "road_type", "light_conditions",
    "weather_conditions", "road_surface_conditions", "junction_detail",
    "junction_control", "first_road_class", "trunk_road_flag",
    "latitude", "longitude", "number_of_vehicles", "number_of_casualties",
]
CASUALTY_COLUMNS = [
    "collision_index", "vehicle_reference", "casualty_reference",
    "casualty_severity", "casualty_class", "age_band_of_casualty",
    "sex_of_casualty", "casualty_type",
]
VEHICLE_COLUMNS = [
    "collision_index", "vehicle_reference", "vehicle_type", "vehicle_manoeuvre",
    "sex_of_driver", "age_band_of_driver", "engine_capacity_cc",
]
TABLE_KEYS = {
    "collisions": ["collision_index"],
    "casualties": ["collision_index", "vehicle_reference", "casualty_reference"],
    "vehicles": ["collision_index", "vehicle_reference"],
}
COLUMN_ALIASES = {"accident_index": "collision_index", "accident_severity": "collision_severity"}


def snake_case(value) -> str:
    """Same column-name standardisation as the notebook (Section 3.4)."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return COLUMN_ALIASES.get(cleaned, cleaned)


def detect_table(columns: list[str]) -> str | None:
    """Recognise a STATS19 table from its columns rather than its filename.

    DfT filenames all contain "road-casualty-statistics", so matching on the
    word "casualty" in a filename would wrongly pick up all three tables.
    """
    names = set(columns)
    if "collision_index" not in names:
        return None
    if names & {"casualty_reference", "casualty_severity"}:
        return "casualties"
    if "collision_severity" in names:
        return "collisions"
    if {"vehicle_reference", "vehicle_type"} <= names:
        return "vehicles"
    return None


def discover_source_files() -> dict[str, list[Path]]:
    """Find every collision, casualty and vehicle CSV, preferring data/."""
    for folder in (DATA_DIR, APP_DIR):
        found: dict[str, list[Path]] = {"collisions": [], "casualties": [], "vehicles": []}
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            try:
                header = pd.read_csv(path, nrows=0).columns
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            table = detect_table([snake_case(column) for column in header])
            if table:
                found[table].append(path)
        if all(found.values()):
            return found
    return found


def data_signature(files: dict[str, list[Path]]) -> tuple:
    """Cache key that changes whenever a data file is added, removed or edited."""
    return tuple(
        (table, path.name, path.stat().st_size, int(path.stat().st_mtime))
        for table, paths in sorted(files.items()) for path in paths
    )


def read_table(paths: list[Path], wanted: list[str], keys: list[str]) -> pd.DataFrame:
    """Read the required columns from every file of one table and combine them."""
    frames = []
    for path in paths:
        raw_header = pd.read_csv(path, nrows=0).columns
        rename = {raw: snake_case(raw) for raw in raw_header}
        usecols = [raw for raw in raw_header if rename[raw] in wanted]
        frames.append(pd.read_csv(path, usecols=usecols, low_memory=False).rename(columns=rename))
    combined = pd.concat(frames, ignore_index=True, sort=False) if len(frames) > 1 else frames[0]
    for column in wanted:
        if column not in combined.columns:
            combined[column] = np.nan
    combined["collision_index"] = combined["collision_index"].astype(str).str.strip()
    if len(frames) > 1:
        # Overlapping exports (for example a five-year file plus an annual file)
        # are de-duplicated on the record key only. Never drop rows on their
        # attribute values: two casualties in one collision can share the same
        # age band, sex and severity and are still two people.
        combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.reset_index(drop=True)


def _small_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int16")


@st.cache_resource(show_spinner="Loading every year of the STATS19 collision, casualty and vehicle data...")
def load_stats19_data(signature: tuple) -> dict:
    files = discover_source_files()
    missing = [table for table, paths in files.items() if not paths]
    if missing:
        raise FileNotFoundError(
            "No CSV was recognised for: " + ", ".join(missing)
            + f". Place the STATS19 CSV files in {DATA_DIR}."
        )

    # ---- Collisions: one row per collision --------------------------------
    raw = read_table(files["collisions"], COLLISION_COLUMNS, TABLE_KEYS["collisions"])
    collisions = pd.DataFrame({"collision_index": raw["collision_index"]})
    collisions["cid"] = np.arange(len(raw), dtype=np.int32)

    date = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")
    if date.isna().mean() > 0.5:  # other export formats
        date = pd.to_datetime(raw["date"], dayfirst=True, errors="coerce")
    collisions["date"] = date
    year = date.dt.year
    if "collision_year" in raw:
        year = year.fillna(pd.to_numeric(raw["collision_year"], errors="coerce"))
    collisions["year"] = year.astype("Int16")
    collisions["month_number"] = date.dt.month.astype("Int8")
    collisions["weekday"] = date.dt.day_name().fillna("Unknown").astype("category")
    collisions["time"] = raw["time"].astype("string").str.strip().fillna("Unknown").astype("category")
    hour = pd.to_numeric(raw["time"].astype("string").str[:2], errors="coerce")
    hour = hour.where(hour.between(0, 23)).astype("Int8")
    collisions["hour"] = hour
    collisions["hour_label"] = (
        (hour.astype("string").str.zfill(2) + ":00").fillna("Unknown").astype("category")
    )

    severity_code = pd.to_numeric(raw["collision_severity"], errors="coerce")
    collisions["severity_label"] = label_codes(severity_code, "collision_severity")
    collisions["is_fatal"] = severity_code.eq(1).to_numpy()
    collisions["is_severe"] = severity_code.isin([1, 2]).to_numpy()

    for source, target in [
        ("urban_or_rural_area", "area_label"), ("road_type", "road_type_label"),
        ("light_conditions", "light_label"), ("weather_conditions", "weather_label"),
        ("road_surface_conditions", "surface_label"),
        ("junction_detail", "junction_detail_label"),
        ("junction_control", "junction_control_label"),
        ("first_road_class", "first_road_class_label"), ("trunk_road_flag", "trunk_label"),
    ]:
        collisions[target] = label_codes(raw[source], source)
    collisions["is_dark"] = pd.to_numeric(raw["light_conditions"], errors="coerce").isin([4, 5, 6, 7]).to_numpy()

    speed = pd.to_numeric(raw["speed_limit"], errors="coerce")
    speed = speed.where(speed > 0).astype("Int16")
    collisions["speed_limit"] = speed
    collisions["speed_label"] = speed.astype("string").fillna("Unknown").astype("category")

    collisions["latitude"] = pd.to_numeric(raw["latitude"], errors="coerce").astype("float32")
    collisions["longitude"] = pd.to_numeric(raw["longitude"], errors="coerce").astype("float32")
    collisions["number_of_vehicles"] = _small_int(raw["number_of_vehicles"])
    collisions["number_of_casualties"] = _small_int(raw["number_of_casualties"])
    del raw

    index_to_cid = pd.Series(collisions["cid"].to_numpy(), index=collisions["collision_index"].to_numpy())

    def attach_cid(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        cid = frame["collision_index"].map(index_to_cid)
        unlinked = int(cid.isna().sum())
        frame = frame.loc[cid.notna()].copy()
        frame.insert(0, "cid", cid.dropna().astype(np.int32).to_numpy())
        return frame.drop(columns="collision_index"), unlinked

    # ---- Casualties: one row per casualty ---------------------------------
    raw = read_table(files["casualties"], CASUALTY_COLUMNS, TABLE_KEYS["casualties"])
    raw, unlinked_casualties = attach_cid(raw)
    casualty_severity = pd.to_numeric(raw["casualty_severity"], errors="coerce")
    casualty_type = pd.to_numeric(raw["casualty_type"], errors="coerce")
    user_type = casualty_type.map(USER_GROUP).astype(object)
    user_type = user_type.where(user_type.notna(), np.where(
        pd.to_numeric(raw["casualty_class"], errors="coerce").eq(3), "Pedestrian", "Other"))
    casualties = pd.DataFrame({
        "cid": raw["cid"].to_numpy(),
        "severity_label": label_codes(casualty_severity, "casualty_severity").values,
        "casualty_class_label": label_codes(raw["casualty_class"], "casualty_class").values,
        "age_band_label": label_codes(raw["age_band_of_casualty"], "age_band").values,
        "sex_label": label_codes(raw["sex_of_casualty"], "sex_of_casualty").values,
        "user_type": pd.Categorical(user_type),
    })
    casualties["is_ksi"] = casualty_severity.isin([1, 2]).to_numpy()
    casualties["is_vru"] = casualties["user_type"].isin(VULNERABLE_USERS).to_numpy()
    del raw

    # ---- Vehicles: one row per vehicle ------------------------------------
    raw = read_table(files["vehicles"], VEHICLE_COLUMNS, TABLE_KEYS["vehicles"])
    raw, unlinked_vehicles = attach_cid(raw)
    engine = pd.to_numeric(raw["engine_capacity_cc"], errors="coerce")
    vehicles = pd.DataFrame({
        "cid": raw["cid"].to_numpy(),
        "vehicle_reference": _small_int(raw["vehicle_reference"]).array,
        "vehicle_type_label": label_codes(raw["vehicle_type"], "vehicle_type").values,
        "vehicle_manoeuvre_label": label_codes(raw["vehicle_manoeuvre"], "vehicle_manoeuvre").values,
        "driver_sex_label": label_codes(raw["sex_of_driver"], "sex_of_driver").values,
        "driver_age_band_label": label_codes(raw["age_band_of_driver"], "age_band").values,
        # Negative codes mean "not recorded", as in notebook Section 6.1.
        "engine_capacity_cc": engine.mask(engine < 0).astype("Float32").array,
    })
    del raw

    return {
        "collisions": collisions,
        "casualties": casualties,
        "vehicles": vehicles,
        "files": {table: [p.name for p in paths] for table, paths in files.items()},
        "unlinked": {"casualties": unlinked_casualties, "vehicles": unlinked_vehicles},
    }


def linked_rows(table: pd.DataFrame, selected: pd.DataFrame, total: int) -> pd.DataFrame:
    """Casualty or vehicle rows that belong to the selected collisions."""
    if len(selected) == total:
        return table
    keep = np.zeros(total, dtype=bool)
    keep[selected["cid"].to_numpy()] = True
    return table[keep[table["cid"].to_numpy()]]


# =============================================================================
# 5. NOTEBOOK OUTPUTS AND MODEL ARTEFACTS
# =============================================================================

@st.cache_data(show_spinner=False)
def _read_output(path: str, modified: float) -> pd.DataFrame:
    return pd.read_csv(path)


def notebook_output(name: str) -> pd.DataFrame | None:
    """Read one evidence table written by the notebook to outputs/."""
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    try:
        return _read_output(str(path), path.stat().st_mtime)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def missing_output(name: str):
    st.info(f"`outputs/{name}` was not found. Run {NOTEBOOK_NAME} to regenerate it.")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def artefact_signature() -> tuple:
    return tuple(
        (p.name, p.stat().st_size, int(p.stat().st_mtime))
        for p in (MODEL_PATH, CONTRACT_PATH, METADATA_PATH) if p.exists()
    )


@st.cache_resource(show_spinner="Loading the saved severity model...")
def load_prediction_assets(signature: tuple):
    """Load the notebook's fitted pipeline and its input contract (Sections 9.2-9.3)."""
    import joblib  # imported here so the dashboards still run without scikit-learn

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Model contract not found: {CONTRACT_PATH}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = joblib.load(MODEL_PATH)
    version_warnings = sorted({
        str(w.message) for w in caught
        if w.category.__name__ == "InconsistentVersionWarning"
    })
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    metadata = load_json(METADATA_PATH)

    specs = []
    for item in contract.get("features") or []:
        if isinstance(item, str):
            specs.append({"feature": item, "dtype": "numeric", "default": 0.0, "category_examples": []})
        elif isinstance(item, dict) and (item.get("feature") or item.get("name")):
            specs.append({**item, "feature": item.get("feature") or item.get("name")})
        else:
            raise ValueError("Unsupported feature definition in model_contract.json.")
    names = [spec["feature"] for spec in specs]
    if not names:
        raise ValueError("model_contract.json must contain a non-empty features list.")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate feature names found in model_contract.json.")
    expected = list(getattr(model, "feature_names_in_", []))
    if expected and expected != names:
        raise ValueError(
            "model_contract.json does not match the saved pipeline's input columns. "
            f"Re-run Sections 9.2 and 9.3 of {NOTEBOOK_NAME}."
        )
    return model, contract, metadata, specs, version_warnings


# =============================================================================
# 6. REUSABLE CHARTS, CARDS AND TABLE HELPERS
# =============================================================================

def panel(title: str, note: str | None = None):
    note_html = f'<p class="panel-note">{note}</p>' if note else ""
    st.markdown(
        f'<div class="panel-heading"><div class="panel-title">{title}</div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def section_heading(title: str):
    st.markdown(f'<div class="section-title" style="margin:0 0 1.15rem">{title}</div>', unsafe_allow_html=True)


def empty_state(message: str):
    st.info(message)


def plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to strings before charting."""
    categorical = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    return frame.astype({c: str for c in categorical}) if categorical else frame


def show_chart(chart, height: int = 340, key: str | None = None):
    chart = chart.properties(height=height, width="container")
    chart = (
        chart.configure(background=PANEL)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, gridColor=BORDER,
            domainColor=BORDER, tickColor=BORDER, labelFont="DM Sans",
            titleFont="DM Sans", labelLimit=260,
        )
        .configure_legend(labelColor=INK, titleColor=MUTED, labelFont="DM Sans")
        .configure_title(color=INK, font="Space Grotesk")
    )
    kwargs = dict(stretch(st.altair_chart))
    if key:
        kwargs["key"] = key
    st.altair_chart(chart, **kwargs)


def show_table(frame: pd.DataFrame, **kwargs):
    st.dataframe(frame, hide_index=True, **stretch(st.dataframe), **kwargs)


def render_cards(cards: list[tuple[str, str, str]], key: str):
    """Compact KPI cards (6 / 3 / 2 columns by screen width)."""
    with st.container(key=key):
        columns = st.columns(len(cards), gap="small")
        for column, (label, value, colour) in zip(columns, cards):
            with column:
                st.markdown(
                    f'<div class="kpi-card" style="background:linear-gradient(145deg,{PANEL_ALT},{PANEL});'
                    f'border:1px solid {BORDER};border-top:3px solid {colour}">'
                    f'<div class="kpi-label" style="color:{MUTED}">{label}</div>'
                    f'<div class="kpi-value" style="color:{INK}">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def severity_scale(domain: list[str] | None = None) -> alt.Scale:
    domain = domain or SEVERITY_ORDER
    return alt.Scale(domain=domain, range=[SEVERITY_COLOURS[x] for x in domain])


def count_frame(series: pd.Series, label="Category", value="Count") -> pd.DataFrame:
    return series.astype("string").fillna("Unknown").value_counts().rename_axis(label).reset_index(name=value)


def pct(value, digits: int = 1) -> str:
    return "–" if value is None or pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def severity_donut(df: pd.DataFrame):
    panel("Collision severity")
    counts = count_frame(df["severity_label"], "Severity", "Collisions")
    counts = counts[counts["Severity"].isin(SEVERITY_ORDER)]
    if counts.empty:
        return empty_state("No records are available for this chart.")
    counts["Share"] = counts["Collisions"] / counts["Collisions"].sum()
    chart = alt.Chart(counts).mark_arc(innerRadius=76, outerRadius=108, stroke=PANEL).encode(
        theta=alt.Theta("Collisions:Q"),
        color=alt.Color("Severity:N", scale=severity_scale(), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Severity:N", alt.Tooltip("Collisions:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    )
    show_chart(chart)


def bar_chart(df, category, value, title, colour=BLUE, horizontal=False, order=None, value_title="Records"):
    panel(title)
    if df.empty:
        return empty_state("No records are available for this chart.")
    if horizontal:
        enc = {"x": alt.X(f"{value}:Q", title=value_title),
               "y": alt.Y(f"{category}:N", sort=order or "-x", title=None)}
    else:
        enc = {"x": alt.X(f"{category}:N", sort=order or "-y", title=None,
                          axis=alt.Axis(labelAngle=-35 if df[category].nunique() > 7 else 0)),
               "y": alt.Y(f"{value}:Q", title=value_title)}
    chart = alt.Chart(plain(df)).mark_bar(cornerRadiusEnd=5, color=colour).encode(
        **enc,
        tooltip=[alt.Tooltip(f"{category}:N", title="Category"),
                 alt.Tooltip(f"{value}:Q", title=value_title, format=",")],
    )
    show_chart(chart)


def stacked_severity_chart(frame, category, title, sort=None, severity_col="severity_label", value_title="Share"):
    """Within-category severity composition (100% stacked bars)."""
    panel(title)
    counts = frame.groupby([category, severity_col], observed=True).size().reset_index(name="Count")
    counts = plain(counts.rename(columns={category: "Category", severity_col: "Severity"}))
    counts = counts[counts["Severity"].isin(SEVERITY_ORDER) & counts["Count"].gt(0)]
    if counts.empty:
        return empty_state("No recognised Fatal, Serious or Slight records are available for this selection.")
    present = set(counts["Category"])
    if sort is None:
        sort = counts.groupby("Category")["Count"].sum().sort_values(ascending=False).index.tolist()
    order = [c for c in sort if c in present]
    counts["Stack"] = counts["Severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
    chart = alt.Chart(counts).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Count:Q", stack="normalize", axis=alt.Axis(format="%"), title=value_title),
        y=alt.Y("Category:N", sort=order, title=None),
        color=alt.Color("Severity:N", sort=SEVERITY_ORDER, scale=severity_scale(),
                        legend=alt.Legend(orient="bottom", title=None)),
        order=alt.Order("Stack:Q"),
        tooltip=["Category:N", "Severity:N", alt.Tooltip("Count:Q", title="Records", format=",")],
    )
    show_chart(chart, height=max(300, min(520, 34 * len(order) + 90)))


def severe_share_table(df: pd.DataFrame, column: str, min_records: int = 250) -> tuple[pd.DataFrame, int]:
    """Serious/Fatal share by category, as in notebook Sections 4.4-4.5."""
    grouped = (
        df.groupby(column, observed=True)
        .agg(Collisions=("is_severe", "size"), Severe=("is_severe", "sum"), Fatal=("is_fatal", "sum"))
        .reset_index().rename(columns={column: "Category"})
    )
    grouped["Category"] = grouped["Category"].astype(str)
    grouped = grouped[grouped["Category"] != "Unknown"]
    hidden = int((grouped["Collisions"] < min_records).sum())
    grouped = grouped[grouped["Collisions"] >= min_records].copy()
    grouped["Severe share (%)"] = grouped["Severe"] / grouped["Collisions"] * 100
    grouped["Fatal per 1,000"] = grouped["Fatal"] / grouped["Collisions"] * 1000
    return grouped.sort_values("Severe share (%)", ascending=False), hidden


def monthly_counts(df):
    valid = df.dropna(subset=["date"])
    if valid.empty:
        return pd.DataFrame(columns=["Year", "MonthNo", "Collisions"])
    return (pd.DataFrame({"Year": valid["year"].astype(int).astype(str), "MonthNo": valid["month_number"].astype(int)})
            .groupby(["Year", "MonthNo"]).size().reset_index(name="Collisions"))


def year_span(years) -> str:
    years = sorted(int(y) for y in years)
    if not years:
        return "no years"
    return str(years[0]) if years[0] == years[-1] else f"{years[0]}–{years[-1]}"


# =============================================================================
# 7. LOAD DATA, SIDEBAR AND GLOBAL FILTERS
# =============================================================================

try:
    _files = discover_source_files()
    DATASET = load_stats19_data(data_signature(_files))
except Exception as exc:
    st.error("The STATS19 datasets could not be loaded.")
    st.exception(exc)
    st.markdown(
        "**Check:** place the collision, casualty and vehicle CSV files in the "
        "`data/` folder beside `App_Prediction.py`. The DfT \"last 5 years\" "
        "files or several annual files can be used; every year found is combined."
    )
    st.stop()

collisions_all: pd.DataFrame = DATASET["collisions"]
casualties_all: pd.DataFrame = DATASET["casualties"]
vehicles_all: pd.DataFrame = DATASET["vehicles"]
TOTAL_COLLISIONS = len(collisions_all)
ALL_YEARS = sorted(int(y) for y in collisions_all["year"].dropna().unique())
if not ALL_YEARS:
    st.error("No valid collision dates were found. Check the date column and CSV format.")
    st.stop()

PAGES = ["Overview", "Collision Analytics", "Road Users", "Statistical Evidence",
         "Severity Prediction", "Model Insights", "Collision Explorer"]
DESCRIPTIVE_PAGES = {"Overview", "Collision Analytics", "Road Users", "Collision Explorer"}

severity_options = [s for s in SEVERITY_ORDER if s in set(collisions_all["severity_label"].cat.categories)]
area_options = sorted(str(c) for c in collisions_all["area_label"].cat.categories)
road_options = sorted(str(c) for c in collisions_all["road_type_label"].cat.categories)
speed_options = sorted((str(c) for c in collisions_all["speed_label"].cat.categories),
                       key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0))
FILTERS = {
    "filter_years": ALL_YEARS,
    "filter_severity": severity_options,
    "filter_area": area_options,
    "filter_road": road_options,
    "filter_speed": speed_options,
}
for _key, _options in FILTERS.items():
    _current = st.session_state.get(_key)
    st.session_state[_key] = list(_options) if _current is None else [v for v in _current if v in _options]


def reset_global_filters():
    for key, options in FILTERS.items():
        st.session_state[key] = list(options)


with st.sidebar:
    st.markdown(
        f'<div style="padding:12px 3px 18px;border-bottom:1px solid {BORDER};margin-bottom:16px">'
        f'<div style="display:flex;align-items:center;gap:10px;font-family:Space Grotesk;font-size:1.24rem;font-weight:800;color:{INK};letter-spacing:-.04em">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;border-radius:11px;background:linear-gradient(145deg,{TEAL},{CYAN});color:#10202B;font-size:1.2rem">🚦</span>'
        f'<span>ROAD<span style="color:{TEAL}">/</span>SAFETY</span></div>'
        f'<div style="color:{MUTED};font-size:.65rem;letter-spacing:.16em;text-transform:uppercase;margin-top:3px">'
        f'Intelligence platform · STATS19 {year_span(ALL_YEARS)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div class='section-kicker'>Workspace</div>", unsafe_allow_html=True)
    page = st.radio("Dashboard section", PAGES, label_visibility="collapsed", key="main_navigation")
    st.divider()
    st.markdown("<div class='section-kicker'>Refine the data</div>", unsafe_allow_html=True)
    st.button("↺  Reset all filters", on_click=reset_global_filters, **stretch(st.button))
    with st.expander("Time period & outcome", expanded=True):
        selected_years = st.multiselect("Collision year", ALL_YEARS, key="filter_years", placeholder="All years")
        selected_severity = st.multiselect("Collision severity", severity_options, key="filter_severity",
                                           placeholder="All severity levels")
    with st.expander("Road context", expanded=False):
        selected_areas = st.multiselect("Area type", area_options, key="filter_area", placeholder="All area types")
        selected_roads = st.multiselect("Road type", road_options, key="filter_road", placeholder="All road types")
        selected_speeds = st.multiselect("Speed limit (mph)", speed_options, key="filter_speed",
                                         placeholder="All speed limits")
    st.markdown(
        f"<div class='filter-caption'>All {len(ALL_YEARS)} years in the data folder are loaded "
        f"({year_span(ALL_YEARS)}). An empty filter means “all”. Filters apply to the Overview, "
        f"Collision Analytics, Road Users and Collision Explorer pages; Statistical Evidence and "
        f"Model Insights show the notebook's results for the full dataset.</div>",
        unsafe_allow_html=True,
    )


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar filters. An empty selection means no restriction."""
    mask = np.ones(len(frame), dtype=bool)
    for column, chosen, options in [
        ("year", selected_years, ALL_YEARS),
        ("severity_label", selected_severity, severity_options),
        ("area_label", selected_areas, area_options),
        ("road_type_label", selected_roads, road_options),
        ("speed_label", selected_speeds, speed_options),
    ]:
        if chosen and set(chosen) != set(options):
            mask &= frame[column].isin(chosen).to_numpy(dtype=bool, na_value=False)
    return frame if mask.all() else frame[mask]


filtered_collisions = apply_filters(collisions_all)
filtered_casualties = linked_rows(casualties_all, filtered_collisions, TOTAL_COLLISIONS)
filtered_vehicles = linked_rows(vehicles_all, filtered_collisions, TOTAL_COLLISIONS)
selected_year_values = sorted(int(y) for y in filtered_collisions["year"].dropna().unique())

st.markdown('<h1 style="margin:0 0 1.4rem 0">Road Safety Intelligence</h1>', unsafe_allow_html=True)
st.markdown(
    f'<div class="coverage">Great Britain · STATS19 · <b>{year_span(ALL_YEARS)}</b> '
    f'({len(ALL_YEARS)} years loaded: {TOTAL_COLLISIONS:,} collisions, {len(casualties_all):,} casualties, '
    f'{len(vehicles_all):,} vehicles) · current selection: <b>{year_span(selected_year_values)}</b>, '
    f'{len(filtered_collisions):,} collisions</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# 8. DESCRIPTIVE PAGES
# =============================================================================

def key_findings(collisions: pd.DataFrame, casualties: pd.DataFrame) -> list[str]:
    """Plain-English statements in the style of notebook Section 4.8."""
    findings = []
    shares = collisions["severity_label"].astype(str).value_counts(normalize=True)
    findings.append(
        "Slight {0}, Serious {1} and Fatal {2} of the {3:,} selected collisions.".format(
            pct(shares.get("Slight", 0)), pct(shares.get("Serious", 0)),
            pct(shares.get("Fatal", 0)), len(collisions))
    )
    speed, _ = severe_share_table(collisions, "speed_label")
    if not speed.empty:
        top = speed.iloc[0]
        findings.append(
            f"Among speed limits with at least 250 collisions, the highest Serious/Fatal share is at "
            f"{top['Category']} mph ({top['Severe share (%)']:.1f}%)."
        )
    area, _ = severe_share_table(collisions, "area_label")
    area = area.set_index("Category")["Severe share (%)"]
    if {"Rural", "Urban"} <= set(area.index):
        findings.append(f"Rural collisions were Serious or Fatal {area['Rural']:.1f}% of the time, "
                        f"against {area['Urban']:.1f}% for urban collisions.")
    light, _ = severe_share_table(collisions, "light_label")
    light = light.set_index("Category")["Severe share (%)"]
    if {"Darkness - no lighting", "Daylight"} <= set(light.index):
        findings.append(f"Collisions in darkness with no street lighting had a Serious/Fatal share of "
                        f"{light['Darkness - no lighting']:.1f}%, against {light['Daylight']:.1f}% in daylight.")
    if not casualties.empty:
        findings.append(f"Pedestrians, cyclists and motorcyclists were {casualties['is_vru'].mean() * 100:.1f}% "
                        f"of the {len(casualties):,} casualties in this selection.")
    return findings


def render_overview():
    section_heading("Overview")
    n_collisions, n_casualties = len(filtered_collisions), len(filtered_casualties)
    casualty_severity = filtered_casualties["severity_label"].astype(str)
    render_cards([
        ("COLLISIONS", f"{n_collisions:,}", BLUE),
        ("CASUALTIES", f"{n_casualties:,}", PURPLE),
        ("FATAL COLLISIONS", f"{int(filtered_collisions['is_fatal'].sum()):,}", RED),
        ("PEOPLE KILLED", f"{int(casualty_severity.eq('Fatal').sum()):,}", CORAL),
        ("SERIOUSLY INJURED", f"{int(casualty_severity.eq('Serious').sum()):,}", AMBER),
        ("VULNERABLE ROAD USERS", f"{filtered_casualties['is_vru'].mean() * 100:.1f}%" if n_casualties else "–", CYAN),
    ], key="kpi-grid-overview")

    items = "".join(f"<li>{text}</li>" for text in key_findings(filtered_collisions, filtered_casualties))
    st.markdown(
        f'<div class="findings"><div class="panel-title">Key findings in this selection</div><ul>{items}</ul>'
        f'<div class="caution">Patterns among recorded collisions. They are not exposure-adjusted risks and '
        f'do not prove causation (notebook Section 4.8).</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.35], gap="large")
    with left:
        severity_donut(filtered_collisions)
    with right:
        panel("Monthly collisions by year")
        trend = monthly_counts(filtered_collisions)
        if trend.empty:
            empty_state("No valid dates in the current selection.")
        else:
            years = sorted(trend["Year"].unique())
            palette = (YEAR_PALETTE * (len(years) // len(YEAR_PALETTE) + 1))[:len(years)]
            chart = alt.Chart(trend).mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=40, filled=True)).encode(
                x=alt.X("MonthNo:O", title=None, sort=list(range(1, 13)),
                        axis=alt.Axis(labelAngle=0, labelExpr=f"{MONTH_ABBR}[datum.value-1]")),
                y=alt.Y("Collisions:Q", title="Collisions", scale=alt.Scale(zero=True)),
                color=alt.Color("Year:N", title="Year", scale=alt.Scale(domain=years, range=palette)),
                tooltip=["Year:N", alt.Tooltip("MonthNo:O", title="Month"), alt.Tooltip("Collisions:Q", format=",")],
            )
            show_chart(chart)

    by_year = (filtered_collisions.dropna(subset=["year"])
               .groupby(["year", "severity_label"], observed=True).size().unstack(fill_value=0))
    by_year.columns = by_year.columns.astype(str)
    left, right = st.columns(2, gap="large")
    with left:
        panel("Collisions by year")
        if by_year.empty:
            empty_state("No valid years in the current selection.")
        else:
            totals = by_year.sum(axis=1).rename("Collisions").reset_index().rename(columns={"year": "Year"})
            totals["Year"] = totals["Year"].astype(int).astype(str)
            chart = alt.Chart(totals).mark_bar(cornerRadiusEnd=5, color=BLUE).encode(
                x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Collisions:Q", title="Collisions"),
                tooltip=["Year:O", alt.Tooltip("Collisions:Q", format=",")],
            )
            show_chart(chart)
    with right:
        panel("Serious and Fatal share of collisions by year")
        if by_year.empty:
            empty_state("No valid years in the current selection.")
        else:
            shares = by_year.div(by_year.sum(axis=1), axis=0) * 100
            shares = shares.reindex(columns=["Serious", "Fatal"], fill_value=0).reset_index()
            shares = shares.rename(columns={"year": "Year"})
            shares["Year"] = shares["Year"].astype(int).astype(str)
            long = shares.melt(id_vars="Year", var_name="Severity", value_name="Share (%)")
            long["Stack"] = long["Severity"].map({"Fatal": 0, "Serious": 1})
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Share (%):Q", title="Share of collisions (%)", stack="zero"),
                color=alt.Color("Severity:N", scale=severity_scale(["Fatal", "Serious"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                order=alt.Order("Stack:Q"),
                tooltip=["Year:O", "Severity:N", alt.Tooltip("Share (%):Q", format=".2f")],
            )
            show_chart(chart)

    left, right = st.columns(2, gap="large")
    with left:
        stacked_severity_chart(filtered_collisions, "area_label", "Severity mix by area")
    with right:
        panel("Collision locations (random sample of up to 5,000)")
        geo = filtered_collisions[["latitude", "longitude", "severity_label"]].dropna(subset=["latitude", "longitude"])
        geo = geo[geo["latitude"].between(49, 61) & geo["longitude"].between(-9, 3)]
        if geo.empty:
            empty_state("No valid UK coordinates are available for this selection.")
        else:
            if len(geo) > 5000:
                geo = geo.sample(5000, random_state=42)
            geo = plain(geo).assign(color=lambda d: d["severity_label"].map(SEVERITY_COLOURS).fillna(BLUE))
            geo["order"] = geo["severity_label"].map({"Slight": 0, "Serious": 1, "Fatal": 2})
            geo = geo.sort_values("order")
            st.map(geo, latitude="latitude", longitude="longitude", color="color", size=60)
            st.caption("Colour = severity: green Slight, amber Serious, red Fatal. Coordinates are for description "
                       "only and are not model inputs (notebook Section 4.9).")


FACTOR_OPTIONS = {
    "Speed limit (mph)": "speed_label",
    "Urban or rural area": "area_label",
    "Road type": "road_type_label",
    "Junction detail": "junction_detail_label",
    "Junction control": "junction_control_label",
    "Light conditions": "light_label",
    "Weather": "weather_label",
    "Road surface": "surface_label",
    "First road class": "first_road_class_label",
    "Trunk road": "trunk_label",
}


def render_collision_analytics():
    section_heading("Collision Analytics")
    panel("Collision patterns by day and hour")
    heat = (filtered_collisions[filtered_collisions["hour_label"].astype(str) != "Unknown"]
            .groupby(["weekday", "hour_label"], observed=True).size().reset_index(name="Collisions"))
    if heat.empty:
        empty_state("No valid day/hour observations.")
    else:
        chart = alt.Chart(plain(heat)).mark_rect(cornerRadius=3).encode(
            x=alt.X("hour_label:O", sort=HOUR_ORDER, title="Hour", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("weekday:O", sort=WEEKDAY_ORDER, title=None),
            color=alt.Color("Collisions:Q", scale=alt.Scale(scheme="tealblues"), title="Collisions"),
            tooltip=[alt.Tooltip("weekday:N", title="Day"), alt.Tooltip("hour_label:N", title="Hour"),
                     alt.Tooltip("Collisions:Q", format=",")],
        )
        show_chart(chart)

    panel("Serious/Fatal share by road and environment factor",
          "Within-category share of collisions that were Serious or Fatal, as in notebook Sections 4.4–4.5. "
          "Categories with fewer than 250 collisions in the current selection are hidden.")
    c1, c2 = st.columns([1, 1])
    with c1:
        factor_name = st.selectbox("Factor", list(FACTOR_OPTIONS), key="analytics_factor")
    with c2:
        measure = st.radio("Measure", ["Serious/Fatal share (%)", "Fatal per 1,000 collisions"],
                           horizontal=True, key="analytics_measure")
    column = FACTOR_OPTIONS[factor_name]
    table, hidden = severe_share_table(filtered_collisions, column)
    if table.empty:
        empty_state("No category has at least 250 collisions in the current selection.")
    else:
        value = "Severe share (%)" if measure.startswith("Serious") else "Fatal per 1,000"
        if column == "speed_label":
            order = sorted(table["Category"], key=lambda x: int(x) if x.isdigit() else 999)
        else:
            order = table.sort_values(value, ascending=False)["Category"].tolist()
        overall = (filtered_collisions["is_severe"].mean() * 100 if value == "Severe share (%)"
                   else filtered_collisions["is_fatal"].mean() * 1000)
        bars = alt.Chart(table).mark_bar(cornerRadiusEnd=4, color=RED if value != "Severe share (%)" else AMBER).encode(
            x=alt.X(f"{value}:Q", title=measure),
            y=alt.Y("Category:N", sort=order, title=None),
            tooltip=["Category:N", alt.Tooltip("Collisions:Q", format=","),
                     alt.Tooltip("Severe:Q", title="Serious or Fatal", format=","),
                     alt.Tooltip("Fatal:Q", format=","),
                     alt.Tooltip("Severe share (%):Q", format=".2f"),
                     alt.Tooltip("Fatal per 1,000:Q", format=".2f")],
        )
        rule = alt.Chart(pd.DataFrame({"overall": [overall]})).mark_rule(color=INK, strokeDash=[4, 4]).encode(
            x="overall:Q", tooltip=[alt.Tooltip("overall:Q", title="Selection average", format=".2f")])
        show_chart(alt.layer(bars, rule), height=max(260, min(520, 36 * len(order) + 80)))
        note = f"Dashed line = average for the current selection ({overall:.2f})."
        if hidden:
            note += f" {hidden} small categor{'y' if hidden == 1 else 'ies'} hidden."
        st.caption(note)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        stacked_severity_chart(filtered_collisions, "light_label", "Severity by light condition")
    with c2:
        stacked_severity_chart(filtered_collisions, "road_type_label", "Severity by road type")


def render_road_users():
    section_heading("Road Users")
    if filtered_casualties.empty:
        return empty_state("No casualty records are linked to the current collision selection.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        users = count_frame(filtered_casualties["user_type"], "Road user", "Casualties")
        bar_chart(users, "Road user", "Casualties", "Casualties by road-user group", colour=TEAL, horizontal=True)
    with c2:
        stacked_severity_chart(filtered_casualties, "age_band_label", "Severity by casualty age band",
                               sort=AGE_BAND_ORDER)
    user_ksi = (filtered_casualties.groupby("user_type", observed=True)
                .agg(Casualties=("is_ksi", "size"), KSI=("is_ksi", "sum")).reset_index())
    user_ksi["KSI share (%)"] = user_ksi["KSI"] / user_ksi["Casualties"] * 100
    user_ksi = plain(user_ksi[user_ksi["Casualties"] >= 50])
    c1, c2 = st.columns(2, gap="large")
    with c1:
        panel("Killed or seriously injured share by road-user group")
        if user_ksi.empty:
            empty_state("Not enough casualty records per road-user group for this comparison.")
        else:
            chart = alt.Chart(user_ksi).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("KSI share (%):Q", title="KSI casualties (%)"),
                y=alt.Y("user_type:N", sort="-x", title=None),
                color=alt.Color("user_type:N", scale=alt.Scale(scheme="tableau10"), legend=None),
                tooltip=[alt.Tooltip("user_type:N", title="Road user"), alt.Tooltip("Casualties:Q", format=","),
                         alt.Tooltip("KSI:Q", format=","), alt.Tooltip("KSI share (%):Q", format=".1f")],
            )
            show_chart(chart)
    with c2:
        stacked_severity_chart(filtered_casualties, "casualty_class_label", "Severity by casualty class")
    st.caption("Casualty fields were analysed separately from the collision model because casualty "
               "severity reveals the collision outcome (notebook cleaning register).")


# =============================================================================
# 9. STATISTICAL EVIDENCE (notebook Sections 3, 5 and 6.2)
# =============================================================================

def residual_category_label(predictor: str, category) -> str:
    if str(category) == "Other or rare":
        return "Other or rare"
    if predictor == "speed_limit":
        try:
            return f"{int(float(category))} mph"
        except (TypeError, ValueError):
            return str(category)
    return code_label(predictor, category)


def render_statistical_evidence():
    section_heading("Statistical Evidence")
    st.caption(f"Results from {NOTEBOOK_NAME} (Sections 3, 5 and 6.2), computed on every collision, vehicle and "
               f"casualty record for {year_span(ALL_YEARS)}. The sidebar filters do not change these results.")

    quality = notebook_output("data_quality_summary.csv")
    quality_lookup = {} if quality is None else dict(zip(quality["Quality area"], quality["Result"]))
    render_cards([
        ("COLLISIONS", f"{TOTAL_COLLISIONS:,}", BLUE),
        ("VEHICLE RECORDS", f"{len(vehicles_all):,}", PURPLE),
        ("CASUALTY RECORDS", f"{len(casualties_all):,}", TEAL),
        ("VEHICLE LINKAGE", str(quality_lookup.get("Vehicle-to-collision linkage", "–")), GREEN),
        ("CASUALTY LINKAGE", str(quality_lookup.get("Casualty-to-collision linkage", "–")), CYAN),
        ("FIELDS WITH −1 CODES", str(quality_lookup.get("Fields containing negative administrative codes", "–")), AMBER),
    ], key="kpi-grid-evidence")

    register = notebook_output("hypothesis_evidence_register.csv")
    panel("Hypotheses H1–H4: association with severity",
          "Chi-square tests with bias-corrected Cramér's V; p-values Benjamini-Hochberg corrected within each family.")
    if register is None:
        missing_output("hypothesis_evidence_register.csv")
    else:
        view = register.copy()
        view["Strongest predictor"] = view["Strongest predictor"].map(feature_label)
        view["Largest Cramér's V"] = view.pop("Largest Cramer V").map(lambda v: f"{v:.4f}")
        show_table(view)

    frames = []
    for name, family in [("H1_road_environment.csv", "H1 Road and environment"),
                         ("H2_driver_vehicle.csv", "H2 Driver and vehicle"),
                         ("H3_temporal.csv", "H3 Temporal"),
                         ("H4_casualty.csv", "H4 Casualty")]:
        table = notebook_output(name)
        if table is not None:
            frames.append(table.assign(Hypothesis=family))
    if frames:
        effects = pd.concat(frames, ignore_index=True)
        effects["Predictor label"] = effects["Predictor"].map(feature_label)
        panel("Effect size of every tested predictor (Cramér's V)",
              "All tested variables were significant after correction; effect sizes are small, which is why "
              "prediction is difficult. H4 uses casualty severity (652,821 casualty records).")
        chart = alt.Chart(effects).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Cramer V:Q", title="Cramér's V (bias-corrected)"),
            y=alt.Y("Predictor label:N", sort="-x", title=None),
            color=alt.Color("Hypothesis:N", scale=alt.Scale(domain=list(HYPOTHESIS_COLOURS),
                                                            range=list(HYPOTHESIS_COLOURS.values())),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Hypothesis:N", alt.Tooltip("Predictor label:N", title="Predictor"),
                     alt.Tooltip("Cramer V:Q", format=".4f"), alt.Tooltip("Chi-square:Q", format=",.1f"),
                     alt.Tooltip("Degrees of freedom:Q"), alt.Tooltip("P-value:Q", format=".2e"),
                     alt.Tooltip("Records:Q", format=",")],
        )
        show_chart(chart, height=max(360, 24 * len(effects) + 80))
    else:
        missing_output("H1_road_environment.csv")

    residuals = notebook_output("standardized_residuals_top_associations.csv")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        panel("Standardised residuals for the three strongest H1 predictors",
              "Positive = more collisions of that severity than expected under independence (notebook Section 5.4).")
        if residuals is None:
            missing_output("standardized_residuals_top_associations.csv")
        else:
            predictors = list(dict.fromkeys(residuals["Predictor"]))
            chosen = st.selectbox("Predictor", predictors, format_func=feature_label, key="evidence_residual_predictor")
            subset = residuals[residuals["Predictor"] == chosen].copy()
            subset["Category label"] = [residual_category_label(chosen, c) for c in subset["Category"]]
            limit = float(subset["Standardized residual"].abs().max())
            base = alt.Chart(subset).encode(
                x=alt.X("severity_label:N", sort=["Slight", "Serious", "Fatal"], title=None,
                        axis=alt.Axis(labelAngle=0, orient="top")),
                y=alt.Y("Category label:N", title=None),
            )
            heat = base.mark_rect(cornerRadius=3).encode(
                color=alt.Color("Standardized residual:Q", title="Residual",
                                scale=alt.Scale(scheme="redblue", reverse=True, domain=[-limit, limit])),
                tooltip=[alt.Tooltip("Category label:N", title="Category"), alt.Tooltip("severity_label:N", title="Severity"),
                         alt.Tooltip("Standardized residual:Q", format="+.2f")],
            )
            text = base.mark_text(fontSize=12, color="#0B1220", fontWeight=600).encode(
                text=alt.Text("Standardized residual:Q", format="+.1f"))
            show_chart(alt.layer(heat, text), height=max(240, 46 * subset["Category label"].nunique() + 40))
    with right:
        numeric = notebook_output("numeric_severity_group_tests.csv")
        pairwise = notebook_output("numeric_pairwise_posthoc_tests.csv")
        panel("Numeric variables across severity groups",
              "Kruskal-Wallis tests (Section 5.5) and corrected pairwise Mann-Whitney tests (Section 5.6).")
        if numeric is None:
            missing_output("numeric_severity_group_tests.csv")
        else:
            view = numeric[["Predictor", "Kruskal-Wallis statistic", "Effect estimate", "Adjusted P-value"]].copy()
            view["Predictor"] = view["Predictor"].map(feature_label)
            view["Kruskal-Wallis statistic"] = view["Kruskal-Wallis statistic"].map(lambda v: f"{v:,.1f}")
            view["Effect estimate"] = view["Effect estimate"].map(lambda v: f"{v:.4f}")
            view["Adjusted P-value"] = view["Adjusted P-value"].map(lambda v: f"{v:.2e}")
            show_table(view)
        if pairwise is not None:
            view = pairwise[["Predictor", "Comparison", "First median", "Second median",
                             "Adjusted P-value", "Significant after correction"]].copy()
            view["Predictor"] = view["Predictor"].map(feature_label)
            view["Adjusted P-value"] = view["Adjusted P-value"].map(lambda v: f"{v:.2e}")
            view["Significant after correction"] = view["Significant after correction"].map({True: "Yes", False: "No"})
            show_table(view)

    with st.expander("Data quality and cleaning decisions (notebook Sections 3.5–3.10)"):
        integrity = notebook_output("structural_integrity_checks.csv")
        cleaning = notebook_output("cleaning_decision_register.csv")
        for frame in (integrity, quality, cleaning):
            if frame is not None:
                show_table(frame.astype(str))
        st.caption("Ordinary blank values occurred only in four coordinate fields (53 records each). "
                   "Negative administrative codes such as −1 were treated as unknown categories or missing "
                   "numeric values, never as measurements.")


# =============================================================================
# 10. CONTRACT-DRIVEN SEVERITY PREDICTION (notebook Sections 6.3-9.3)
# =============================================================================

PREDICTION_GROUPS = [
    ("Road and location", [
        "road_type", "speed_limit", "urban_or_rural_area", "trunk_road_flag",
        "first_road_class", "second_road_class", "junction_detail", "junction_control",
        "light_conditions", "weather_conditions", "road_surface_conditions",
        "special_conditions_at_site", "carriageway_hazards",
    ]),
    ("Date and time", ["day_of_week", "month", "hour"]),
    ("Vehicles and drivers", [
        "number_of_vehicles", "vehicle_type_first", "vehicle_manoeuvre_first",
        "driver_sex_first", "driver_age_mean", "driver_age_min",
        "engine_cc_mean", "vehicle_age_mean",
    ]),
]
# Derived exactly as in the notebook so that the inputs stay consistent.
DERIVED_FEATURES = {
    "time_band": "derived from the hour (Section 4.1 bins)",
    "weekend_flag": "derived from the day of week (Saturday or Sunday)",
    "vehicle_records_count": "equal to the number of vehicles (true for every 2021–2025 collision)",
}
NUMERIC_INPUTS = {
    "number_of_vehicles": {"min_value": 1, "max_value": 30, "step": 1, "blank": False},
    "driver_age_mean": {"min_value": 1.0, "max_value": 105.0, "step": 1.0, "blank": True},
    "driver_age_min": {"min_value": 1.0, "max_value": 105.0, "step": 1.0, "blank": True},
    "engine_cc_mean": {"min_value": 0.0, "max_value": 20000.0, "step": 50.0, "blank": True},
    "vehicle_age_mean": {"min_value": 0.0, "max_value": 100.0, "step": 1.0, "blank": True},
}
SPEED_LIMITS = [20, 30, 40, 50, 60, 70]


def time_band_for(hour: float) -> str:
    """Same bins as notebook Section 4.1: [-1, 5, 9, 15, 19, 23]."""
    hour = int(hour)
    if hour <= 5:
        return "Night"
    if hour <= 9:
        return "Morning peak"
    if hour <= 15:
        return "Daytime"
    if hour <= 19:
        return "Evening peak"
    return "Late evening"


def ordered_codes(spec: dict) -> list[str]:
    """Every category the model saw, in a readable order (unknown codes last)."""
    codes = [str(code) for code in spec.get("category_examples") or []]
    default = str(spec.get("default"))
    if default not in codes:
        codes.insert(0, default)

    def sort_key(code: str):
        try:
            value = int(float(code))
        except (TypeError, ValueError):
            return (2, 0, code)
        if spec["feature"] == "day_of_week":
            return (0, (value + 5) % 7, code)  # Monday first
        return (1 if value < 0 or value == 99 else 0, value, code)

    return sorted(dict.fromkeys(codes), key=sort_key)


def prediction_widget(spec: dict, important: set[str]):
    """Render one input and return it in the representation the pipeline expects."""
    name = spec["feature"]
    label = feature_label(name) + (" ★" if name in important else "")
    key = f"predict_{name}"
    default = spec.get("default")
    numeric = str(spec.get("dtype", "")).lower() in {"numeric", "number", "float", "integer", "int"}

    if name == "speed_limit":
        try:
            index = SPEED_LIMITS.index(int(float(default)))
        except (TypeError, ValueError):
            index = 1
        return float(st.selectbox(label, SPEED_LIMITS, index=index, key=key))

    if name == "hour":
        try:
            index = min(max(int(float(default)), 0), 23)
        except (TypeError, ValueError):
            index = 12
        return float(st.selectbox(label, list(range(24)), index=index, format_func=lambda h: f"{h:02d}:00", key=key))

    if name in NUMERIC_INPUTS:
        rules = NUMERIC_INPUTS[name]
        try:
            value = float(default)
        except (TypeError, ValueError):
            value = float(rules["min_value"])
        value = min(max(value, float(rules["min_value"])), float(rules["max_value"]))
        if not rules["blank"]:
            return float(st.number_input(label, min_value=int(rules["min_value"]), max_value=int(rules["max_value"]),
                                         value=int(round(value)), step=1, key=key))
        return float(st.number_input(label, min_value=rules["min_value"], max_value=rules["max_value"],
                                     value=value, step=rules["step"], key=key))

    if not numeric:
        codes = ordered_codes(spec)
        index = codes.index(str(default)) if str(default) in codes else 0
        has_labels = FIELD_ALIASES.get(name, name) in STATS19_LABELS
        return st.selectbox(
            label, codes, index=index, key=key,
            format_func=(lambda code, field=name: code_label(field, code)) if has_labels else str,
        )

    try:
        value = float(default)
    except (TypeError, ValueError):
        value = 0.0
    return float(st.number_input(label, value=value, key=key))


def readable_input(name: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not recorded (imputed with the training median)"
    if name == "speed_limit":
        return f"{value:g} mph"
    if name == "hour":
        return f"{int(value):02d}:00"
    if name == "weekend_flag":
        return "Weekend" if float(value) == 1 else "Weekday"
    if isinstance(value, str) and FIELD_ALIASES.get(name, name) in STATS19_LABELS:
        return code_label(name, value)
    if isinstance(value, float):
        return f"{value:,.1f}".rstrip("0").rstrip(".")
    return str(value)


def render_prediction():
    section_heading("Severity Prediction")
    st.markdown(f'<div class="notice">{RESEARCH_ONLY_NOTICE}</div>', unsafe_allow_html=True)
    try:
        model, contract, metadata, specs, version_warnings = load_prediction_assets(artefact_signature())
    except ModuleNotFoundError as exc:
        st.error(f"A Python package needed by the saved model is missing: {exc.name}.")
        st.code(f'"{sys.executable}" -m pip install -r requirements.txt', language="bash")
        st.caption("The pipeline was saved with scikit-learn 1.7.2; requirements.txt pins that version.")
        return
    except Exception as exc:
        st.error("The prediction assets could not be loaded.")
        st.exception(exc)
        st.info("Confirm that stats19_severity_model.joblib, model_contract.json and metadata.json are in models/.")
        return
    for message in version_warnings:
        st.warning("The model was saved with a different scikit-learn version than this app is running. "
                   "Install the version in requirements.txt (scikit-learn==1.7.2). Details: " + message)

    names = [spec["feature"] for spec in specs]
    spec_by_name = {spec["feature"]: spec for spec in specs}
    importance = notebook_output("permutation_importance.csv")
    important = set(importance.head(10)["Feature"]) if importance is not None else set()
    model_name = metadata.get("model", "the selected model")
    st.caption(
        f"{model_name} pipeline from {NOTEBOOK_NAME}, trained on 411,040 collisions (2021–2025) with "
        f"{len(names)} predictors. It estimates the severity of a collision that has already happened; "
        f"it does not predict whether a collision will occur. ★ = one of the ten most influential inputs "
        f"(permutation importance, Section 8.4). Defaults are the training medians and most common categories."
    )

    grouped = {name for _, members in PREDICTION_GROUPS for name in members}
    groups = list(PREDICTION_GROUPS)
    others = [n for n in names if n not in grouped and n not in DERIVED_FEATURES]
    if others:
        groups.append(("Other model inputs", others))

    values: dict[str, Any] = {}
    with st.form("collision_severity_prediction"):
        for title, members in groups:
            members = [m for m in members if m in spec_by_name]
            if not members:
                continue
            st.markdown(f"<div class='form-group-title'>{title}</div>", unsafe_allow_html=True)
            columns = st.columns(3)
            for i, name in enumerate(members):
                with columns[i % 3]:
                    values[name] = prediction_widget(spec_by_name[name], important)
        blankable = [n for n, rules in NUMERIC_INPUTS.items() if rules["blank"] and n in spec_by_name]
        not_recorded = st.multiselect(
            "Values not recorded for this collision", blankable, format_func=feature_label,
            key="predict_not_recorded", placeholder="All values recorded",
            help="Selected values are sent as missing, and the pipeline imputes the training median, "
                 "exactly as it did for missing values during training.",
        ) if blankable else []
        derived_notes = [f"{feature_label(n)} is {why}" for n, why in DERIVED_FEATURES.items() if n in spec_by_name]
        if derived_notes:
            st.caption("; ".join(derived_notes) + ".")
        submitted = st.form_submit_button("Run severity prediction", type="primary", **stretch(st.form_submit_button))

    if not submitted:
        return

    for name in not_recorded:
        values[name] = np.nan
    if "time_band" in spec_by_name:
        values["time_band"] = time_band_for(values["hour"]) if "hour" in values else str(spec_by_name["time_band"]["default"])
    if "weekend_flag" in spec_by_name:
        values["weekend_flag"] = (1.0 if str(values.get("day_of_week")) in {"1", "7"} else 0.0) \
            if "day_of_week" in values else float(spec_by_name["weekend_flag"]["default"])
    if "vehicle_records_count" in spec_by_name:
        values["vehicle_records_count"] = float(values.get("number_of_vehicles", spec_by_name["vehicle_records_count"]["default"]))

    youngest, mean_age = values.get("driver_age_min"), values.get("driver_age_mean")
    if youngest is not None and mean_age is not None and not np.isnan(youngest) and not np.isnan(mean_age) \
            and youngest > mean_age:
        st.error("The youngest driver's age cannot be greater than the mean driver age. Adjust the inputs and run again.")
        return

    try:
        row = {}
        for spec in specs:
            name, value = spec["feature"], values[spec["feature"]]
            if str(spec.get("dtype", "")).lower() in {"numeric", "number", "float", "integer", "int"}:
                row[name] = np.nan if value is None else float(value)
            else:
                row[name] = str(value)  # the notebook encoded every category as a string
        input_frame = pd.DataFrame([row], columns=names)
        prediction = str(np.asarray(model.predict(input_frame)).reshape(-1)[0])
        classes = [str(c) for c in getattr(model, "classes_", contract.get("classes", []))]
        scores = np.asarray(model.predict_proba(input_frame))[0] if hasattr(model, "predict_proba") else None
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")
        st.exception(exc)
        return

    st.markdown("---")
    colour = SEVERITY_COLOURS.get(prediction, BLUE)
    diagnostics = notebook_output("class_level_diagnostics.csv")
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown(
            f'<div class="result-card" style="border-left:5px solid {colour}">'
            f'<div class="result-label">Predicted collision severity</div>'
            f'<div class="result-value" style="color:{colour}">{prediction}</div>'
            f'<div class="small-muted">Research model output · not an operational assessment</div></div>',
            unsafe_allow_html=True,
        )
        panel("How reliable is this kind of prediction?")
        if diagnostics is not None and prediction in set(diagnostics["Severity"]):
            row_d = diagnostics.set_index("Severity").loc[prediction]
            test_size = int(diagnostics["Actual cases"].sum())
            st.markdown(
                f"On the notebook's untouched test set ({test_size:,} collisions), **{prediction}** predictions "
                f"were correct **{pct(row_d['Precision'])}** of the time (precision), and the model found "
                f"**{pct(row_d['Recall'])}** of the actual {prediction} collisions (recall)."
            )
        base_rate = collisions_all["severity_label"].astype(str).eq(prediction).mean()
        st.markdown(f"For context, {pct(base_rate)} of all {TOTAL_COLLISIONS:,} recorded collisions in "
                    f"{year_span(ALL_YEARS)} were {prediction}.")
    with right:
        if scores is None or len(scores) != len(classes):
            st.info("Class scores are unavailable for this estimator.")
        else:
            score_frame = pd.DataFrame({"Severity": classes, "Score (%)": scores.astype(float) * 100})
            panel("Class scores",
                  "The network was trained with balanced class weights, so Serious and Fatal scores are "
                  "deliberately inflated relative to how often they occur. Read the scores as a ranking "
                  "signal, not as calibrated probabilities.")
            chart = alt.Chart(score_frame).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X("Score (%):Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Severity:N", sort=SEVERITY_ORDER, title=None),
                color=alt.Color("Severity:N", scale=severity_scale(), legend=None),
                tooltip=["Severity:N", alt.Tooltip("Score (%):Q", format=".2f")],
            )
            show_chart(chart, height=220)

    with st.expander("Review the exact inputs sent to the model"):
        review = pd.DataFrame({
            "Feature": [feature_label(n) for n in names],
            "Model column": names,
            "Value sent": ["NaN" if isinstance(row[n], float) and np.isnan(row[n]) else str(row[n]) for n in names],
            "Meaning": [readable_input(n, row[n]) for n in names],
        })
        show_table(review)
    with st.expander("Prediction scope and limitations"):
        st.write(contract.get("scope_warning", "The model predicts severity conditional on a recorded collision."))
        st.warning(f"Notebook acceptance decision: {contract.get('operational_warning', 'not recorded')}. "
                   + RESEARCH_ONLY_NOTICE)
        st.write("The test-set results on the Model Insights page show the error rates to expect, including "
                 "missed severe collisions and false alarms.")


# =============================================================================
# 11. MODEL INSIGHTS (notebook Sections 7-9)
# =============================================================================

def hypotheses_h5_to_h7() -> pd.DataFrame | None:
    """Re-derive the notebook's H5-H7 conclusions from its saved tables (Section 10.3)."""
    cv = notebook_output("model_comparison.csv")
    register = notebook_output("model_register.csv")
    if cv is None or register is None or "Role" not in cv:
        return None
    baseline = cv[cv["Role"] == "Baseline"]
    candidates = cv[cv["Role"] == "Candidate"]
    if baseline.empty or candidates.empty:
        return None
    base = baseline.iloc[0]
    rows = []
    beats = (candidates["Macro-F1"] > base["Macro-F1"]) & (candidates["Balanced accuracy"] > base["Balanced accuracy"])
    rows.append({
        "Hypothesis": "H5 Predictive information beyond a no-skill model",
        "Evidence": f"{int(beats.sum())} of {len(candidates)} candidates beat the baseline's CV Macro-F1 "
                    f"({base['Macro-F1']:.3f}) and balanced accuracy ({base['Balanced accuracy']:.3f}).",
        "Result": "Supported" if beats.all() else ("Partly supported" if beats.any() else "Not supported"),
    })
    handling = register.set_index("Model")["Class-imbalance handling"].astype(str)
    candidate_models = candidates.set_index("Model")
    unweighted = [m for m in candidate_models.index if handling.get(m, "").startswith("None")]
    weighted = [m for m in candidate_models.index if m not in unweighted]
    if weighted and unweighted:
        w = candidate_models.loc[weighted, "Balanced accuracy"]
        u = candidate_models.loc[unweighted, "Balanced accuracy"]
        rows.append({
            "Hypothesis": "H6 Class weighting improves balanced performance",
            "Evidence": f"Weighted candidates: CV balanced accuracy {w.min():.3f}–{w.max():.3f}; "
                        f"unweighted ({', '.join(unweighted)}): {u.max():.3f}.",
            "Result": "Supported" if w.min() > u.max() else "Not supported",
        })
    linear = "Weighted logistic"
    if linear in candidate_models.index:
        linear_f1 = candidate_models.loc[linear, "Macro-F1"]
        nonlinear = candidate_models.drop(index=linear)
        better = nonlinear[nonlinear["Macro-F1"] > linear_f1].index.tolist()
        rows.append({
            "Hypothesis": "H7 Nonlinear models add value beyond the linear benchmark",
            "Evidence": f"CV Macro-F1 above weighted logistic ({linear_f1:.3f}): "
                        + (", ".join(better) if better else "none") + ".",
            "Result": "Supported" if len(better) == len(nonlinear) else
                      ("Supported for " + " and ".join(better) if better else "Not supported"),
        })
    return pd.DataFrame(rows)


def render_model_insights():
    section_heading("Model Insights")
    metadata = load_json(METADATA_PATH)
    contract = load_json(CONTRACT_PATH)
    selected_model = metadata.get("model", "Weighted neural network")
    gate = notebook_output("model_acceptance_gate.csv")
    decision = metadata.get("acceptance_decision") or contract.get("operational_warning", "Not recorded")
    passed_text = ""
    if gate is not None:
        passed = gate["Passed"].astype(str).str.lower().eq("true")
        passed_text = f" — passed {int(passed.sum())} of {len(gate)} acceptance tests"
    st.markdown(
        f'<div class="notice"><b>Selected model:</b> {selected_model} (highest three-fold cross-validated '
        f'Macro-F1). <b>Acceptance decision:</b> {decision}{passed_text}. {RESEARCH_ONLY_NOTICE}</div>',
        unsafe_allow_html=True,
    )

    final = notebook_output("final_test_metrics.csv")
    metrics = final.iloc[0].to_dict() if final is not None and not final.empty else metadata.get("metrics", {})
    if metrics:
        render_cards([
            ("TEST MACRO-F1", f"{metrics.get('Macro-F1', float('nan')):.3f}", TEAL),
            ("BALANCED ACCURACY", pct(metrics.get("Balanced accuracy")), BLUE),
            ("OVERALL ACCURACY", pct(metrics.get("Overall accuracy")), PURPLE),
            ("WEIGHTED F1", f"{metrics.get('Weighted F1', float('nan')):.3f}", CYAN),
            ("SERIOUS RECALL", pct(metrics.get("Serious recall")), AMBER),
            ("FATAL RECALL", pct(metrics.get("Fatal recall")), RED),
        ], key="kpi-grid-model")
        st.caption("Measured once on the untouched 20% test set (102,761 collisions). The no-skill baseline "
                   "that always predicts Slight reaches 75.8% accuracy but only 0.287 Macro-F1, which is why "
                   "accuracy was not used to choose the model.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        panel("Acceptance gate (Section 9.1)")
        if gate is None:
            missing_output("model_acceptance_gate.csv")
        else:
            view = gate[["Acceptance test", "Passed"]].copy()
            view["Passed"] = view["Passed"].astype(str).str.lower().map({"true": "✅ Passed", "false": "❌ Not met"})
            show_table(view)
        panel("Hypotheses H5–H7 (Section 10.3)")
        h_rows = hypotheses_h5_to_h7()
        if h_rows is None:
            missing_output("model_comparison.csv")
        else:
            show_table(h_rows)
    with right:
        comparison = notebook_output("candidate_model_test_comparison.csv")
        panel("Candidate models on the test set (Section 8.1)")
        if comparison is None:
            missing_output("candidate_model_test_comparison.csv")
        else:
            measures = ["Test Macro-F1", "Balanced accuracy", "Serious recall", "Fatal recall"]
            long = comparison.melt(id_vars=["Model"], value_vars=measures, var_name="Measure", value_name="Value")
            order = comparison.sort_values("CV rank")["Model"].tolist()
            chart = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("Model:N", sort=order, title=None),
                yOffset=alt.YOffset("Measure:N", sort=measures),
                x=alt.X("Value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Measure:N", sort=measures, scale=alt.Scale(range=[TEAL, BLUE, AMBER, RED]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Model:N", "Measure:N", alt.Tooltip("Value:Q", format=".3f")],
            )
            show_chart(chart, height=380)

    if comparison is not None:
        with st.expander("Full candidate comparison: cross-validation and test results"):
            view = comparison.copy()
            for column in ["CV Macro-F1", "Test Macro-F1", "Balanced accuracy", "Overall accuracy", "Weighted F1"]:
                view[column] = view[column].map(lambda v: f"{v:.3f}")
            for column in ["Slight recall", "Serious recall", "Fatal recall", "Severe predicted Slight"]:
                view[column] = view[column].map(pct)
            view["Fit time (seconds)"] = view["Fit time (seconds)"].map(lambda v: f"{v:,.1f}")
            show_table(view)
            cv = notebook_output("model_comparison.csv")
            if cv is not None:
                st.caption("Three-fold stratified cross-validation on 120,000 training records (Section 7.4), "
                           "including the no-skill baseline:")
                cv_view = cv.drop(columns=[c for c in ["Run time (seconds)"] if c in cv]).copy()
                for column in ["Macro-F1", "Macro-F1 variation", "Balanced accuracy", "Weighted F1", "Overall accuracy"]:
                    if column in cv_view:
                        cv_view[column] = cv_view[column].map(lambda v: f"{v:.3f}")
                show_table(cv_view)

    summary = notebook_output("confusion_matrix_summary_all_models.csv")
    matrices = notebook_output("confusion_matrices_all_models.csv")
    left, right = st.columns(2, gap="large")
    with left:
        panel("Trade-off: missed severe collisions vs false alarms (Section 8.3)",
              "Lower-left is better. Models that rarely raise false alarms miss most Serious and Fatal collisions.")
        if summary is None:
            missing_output("confusion_matrix_summary_all_models.csv")
        else:
            data = summary.assign(Selected=summary["Model"].eq(selected_model).map({True: "Selected", False: "Other"}))
            base = alt.Chart(data).encode(
                x=alt.X("False-alarm rate (%):Q", title="Slight collisions flagged as severe (%)", scale=alt.Scale(domain=[0, 60])),
                y=alt.Y("Severe missed (%):Q", title="Serious/Fatal predicted Slight (%)", scale=alt.Scale(domain=[0, 100])),
            )
            points = base.mark_circle(size=220, opacity=0.95).encode(
                color=alt.Color("Selected:N", scale=alt.Scale(domain=["Selected", "Other"], range=[TEAL, MUTED]), legend=None),
                tooltip=["Model:N", alt.Tooltip("Severe missed (%):Q", format=".1f"),
                         alt.Tooltip("False-alarm rate (%):Q", format=".1f"),
                         alt.Tooltip("Fatal precision (%):Q", format=".1f"),
                         alt.Tooltip("Correct predictions (%):Q", format=".1f")],
            )
            labels = base.mark_text(align="left", dx=12, color=INK, fontSize=11).encode(text="Model:N")
            show_chart(alt.layer(points, labels), height=360)
    with right:
        panel("Confusion matrix (untouched test set)")
        if matrices is None:
            missing_output("confusion_matrices_all_models.csv")
        else:
            models = list(dict.fromkeys(matrices["Model"]))
            index = models.index(selected_model) if selected_model in models else 0
            chosen = st.selectbox("Model", models, index=index, key="insights_confusion_model",
                                  label_visibility="collapsed")
            cm = matrices[matrices["Model"] == chosen].melt(id_vars=["Model", "Actual"], var_name="Predicted", value_name="Count")
            cm["Actual"] = cm["Actual"].str.replace("Actual ", "", regex=False)
            cm["Predicted"] = cm["Predicted"].str.replace("Predicted ", "", regex=False)
            cm["Row share"] = cm["Count"] / cm.groupby("Actual")["Count"].transform("sum")
            cm["Label"] = cm.apply(lambda r: f"{int(r['Count']):,}\n({r['Row share']:.1%})", axis=1)
            order = ["Slight", "Serious", "Fatal"]
            base = alt.Chart(cm).encode(
                x=alt.X("Predicted:N", sort=order, title="Predicted severity", axis=alt.Axis(labelAngle=0, orient="top")),
                y=alt.Y("Actual:N", sort=order, title="Actual severity"),
            )
            heat = base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("Row share:Q", scale=alt.Scale(scheme="tealblues", domain=[0, 1]), legend=None),
                tooltip=["Actual:N", "Predicted:N", alt.Tooltip("Count:Q", format=","), alt.Tooltip("Row share:Q", format=".1%")],
            )
            text = base.mark_text(lineBreak="\n", fontSize=13, fontWeight=600).encode(
                text="Label:N",
                color=alt.condition("datum['Row share'] > 0.5", alt.value("#0B1220"), alt.value(INK)),
            )
            show_chart(alt.layer(heat, text), height=300)
            st.caption("Each cell shows the count and the share of that actual class; the diagonal is correct.")

    diagnostics = notebook_output("class_level_diagnostics_all_models.csv")
    if diagnostics is not None:
        with st.expander("Class-level precision, recall and F1 for every model (Section 8.2)"):
            view = diagnostics.copy()
            for column in ["Precision", "Recall", "Miss rate"]:
                view[column] = view[column].map(pct)
            view["F1 score"] = view["F1 score"].map(lambda v: f"{v:.3f}")
            show_table(view)

    importance = notebook_output("permutation_importance.csv")
    management = notebook_output("management_interpretation_matrix.csv")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        panel("Permutation importance of the selected model (Section 8.4)",
              "Drop in test Macro-F1 when a predictor is shuffled (15,000 test records, three repeats). "
              "Whiskers = ± one standard deviation; grey bars are within their own variation.")
        if importance is None:
            missing_output("permutation_importance.csv")
        else:
            data = importance.copy()
            data["Predictor"] = data["Feature"].map(feature_label)
            data["Low"] = data["Mean importance"] - data["Importance variation"]
            data["High"] = data["Mean importance"] + data["Importance variation"]
            data["Signal"] = np.where(data["Mean importance"] > data["Importance variation"],
                                      "Larger than its variation", "Within its variation")
            order = data.sort_values("Mean importance", ascending=False)["Predictor"].tolist()
            bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("Mean importance:Q", title="Mean drop in Macro-F1"),
                y=alt.Y("Predictor:N", sort=order, title=None),
                color=alt.Color("Signal:N", scale=alt.Scale(domain=["Larger than its variation", "Within its variation"],
                                                            range=[TEAL, "#56657A"]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Predictor:N", alt.Tooltip("Mean importance:Q", format=".4f"),
                         alt.Tooltip("Importance variation:Q", format=".4f")],
            )
            whiskers = alt.Chart(data).mark_rule(color=INK, opacity=0.6).encode(
                x="Low:Q", x2="High:Q", y=alt.Y("Predictor:N", sort=order))
            show_chart(alt.layer(bars, whiskers), height=max(420, 20 * len(data) + 80))
    with right:
        panel("Management interpretation matrix (Section 8.4A)")
        if management is None:
            missing_output("management_interpretation_matrix.csv")
        else:
            view = management[["Feature", "Mean importance", "Management review area"]].copy()
            view["Feature"] = view["Feature"].map(feature_label)
            view["Mean importance"] = view["Mean importance"].map(lambda v: f"{v:.4f}")
            show_table(view)
            st.caption("Predictive reliance is not evidence that a feature caused severity.")

        panel("Latest-year robustness: train 2021–2024, test 2025 (Section 8.6)")
        holdout = notebook_output("temporal_holdout_metrics.csv")
        if holdout is None or final is None:
            missing_output("temporal_holdout_metrics.csv")
        else:
            h = holdout.iloc[0]
            f = final.iloc[0]
            view = pd.DataFrame({
                "Measure": ["Macro-F1", "Balanced accuracy", "Serious recall", "Fatal recall"],
                f"{int(h['Test year'])} holdout": [f"{h['Macro-F1']:.3f}", f"{h['Balanced accuracy']:.3f}",
                                                   pct(h["Serious recall"]), pct(h["Fatal recall"])],
                "Random test split": [f"{f['Macro-F1']:.3f}", f"{f['Balanced accuracy']:.3f}",
                                      pct(f["Serious recall"]), pct(f["Fatal recall"])],
            })
            show_table(view)
            st.caption(f"Trained on {int(h['Training records']):,} earlier collisions and tested on "
                       f"{int(h['Test records']):,} from {int(h['Test year'])}: performance stayed close to the random split.")

    with st.expander("Missed severe collisions: where the model fails (Sections 8.5–8.5A)"):
        numeric_profile = notebook_output("severe_error_numeric_profile.csv")
        if numeric_profile is not None:
            view = numeric_profile.copy()
            view["Feature"] = view["Feature"].map(feature_label)
            view["mean"] = view["mean"].map(lambda v: f"{v:.2f}")
            show_table(view.rename(columns={"count": "Cases", "mean": "Mean", "median": "Median"}))
        categorical_profile = notebook_output("severe_error_categorical_profile.csv")
        if categorical_profile is not None:
            view = categorical_profile.copy()
            view["Category"] = [code_label(feat, cat) for feat, cat in zip(view["Feature"], view["Category"])]
            view["Feature"] = view["Feature"].map(feature_label)
            show_table(view)
        st.caption("Missed severe cases were concentrated in urban collisions and collisions where the first "
                   "vehicle was a car; collisions in unlit darkness were rarely missed.")

    with st.expander("Model artefacts and limitations"):
        status = pd.DataFrame([
            {"Artefact": "Trained pipeline", "File": "models/" + MODEL_PATH.name, "Available": "Yes" if MODEL_PATH.exists() else "No"},
            {"Artefact": "Input contract", "File": "models/" + CONTRACT_PATH.name, "Available": "Yes" if CONTRACT_PATH.exists() else "No"},
            {"Artefact": "Evaluation metadata", "File": "models/" + METADATA_PATH.name, "Available": "Yes" if METADATA_PATH.exists() else "No"},
        ])
        show_table(status)
        warnings_table = notebook_output("model_warnings.csv")
        if warnings_table is not None and not warnings_table.empty:
            st.caption("Training warnings recorded by the notebook: "
                       + "; ".join(sorted(set(warnings_table["Message"].astype(str)))))
        st.markdown(
            "- **Recorded collisions only.** Descriptive results are not exposure-adjusted risks and do not prove causation.\n"
            "- **Predictive reliance is not causation.** Importance shows what the model relies on, not what causes severity.\n"
            "- **Acceptance gate not met.** Macro-F1, balanced accuracy, Serious recall and Fatal recall are all below 0.60; "
            "about a third of severe test collisions were predicted Slight.\n"
            "- **First vehicle only.** Vehicle information is summarised from first-vehicle attributes and averages.\n"
            "- **Scope.** " + contract.get("scope_warning", "Severity is predicted conditional on a recorded collision.")
        )


# =============================================================================
# 12. COLLISION EXPLORER
# =============================================================================

def render_collision_explorer():
    section_heading("Collision Explorer")
    records = filtered_collisions
    left, right = st.columns([2, 1])
    with left:
        query = st.text_input("Search collision ID, road type or area", key="explorer_search")
    with right:
        page_size = st.selectbox("Rows per page", [25, 50, 100, 250], index=0, key="explorer_page_size")

    if query.strip():
        q = query.strip().lower()
        mask = (
            records["collision_index"].str.lower().str.contains(q, na=False, regex=False)
            | records["road_type_label"].astype(str).str.lower().str.contains(q, na=False, regex=False)
            | records["area_label"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        )
        records = records[mask]

    present = set(records["severity_label"].astype(str))
    severity_filter = st.multiselect("Narrow by severity", SEVERITY_ORDER,
                                     default=[s for s in SEVERITY_ORDER if s in present], key="explorer_severity")
    if severity_filter:
        records = records[records["severity_label"].astype(str).isin(severity_filter)]
    if records.empty:
        return empty_state("No records match the explorer search and filters.")
    records = records.sort_values("date", ascending=False)

    total_pages = max(1, int(np.ceil(len(records) / page_size)))
    page_number = st.number_input(f"Page (of {total_pages:,})", min_value=1, max_value=total_pages, value=1, step=1)
    start = (int(page_number) - 1) * page_size
    current = records.iloc[start:start + page_size]

    display = pd.DataFrame({
        "Collision ID": current["collision_index"],
        "Date": current["date"].dt.strftime("%d %b %Y"),
        "Time": current["time"].astype(str),
        "Severity": current["severity_label"].astype(str),
        "Vehicles": current["number_of_vehicles"],
        "Casualties": current["number_of_casualties"],
        "Speed limit (mph)": current["speed_label"].astype(str),
        "Road type": current["road_type_label"].astype(str),
        "Area": current["area_label"].astype(str),
        "Lighting": current["light_label"].astype(str),
        "Weather": current["weather_label"].astype(str),
    })
    st.caption(f"{len(records):,} matching collisions · showing {start + 1:,}–{start + len(current):,}")
    show_table(display)

    selected_id = st.selectbox("Select a collision to inspect", current["collision_index"].tolist(),
                               key="explorer_collision_id")
    if selected_id:
        record = current[current["collision_index"] == selected_id].iloc[0]
        st.markdown(f"### Collision {selected_id}")
        date_text = record["date"].strftime("%d %b %Y") if pd.notna(record["date"]) else "Date unavailable"
        st.markdown(
            f"**{record['severity_label']}** · {date_text} {record['time']} · {record['area_label']} · "
            f"{record['road_type_label']} · {record['speed_label']} mph · {record['light_label']} · "
            f"{record['weather_label']} · {record['surface_label']}"
        )
        v1, v2 = st.columns(2, gap="large")
        with v1:
            panel("Vehicles involved")
            linked = vehicles_all[vehicles_all["cid"] == record["cid"]]
            if linked.empty:
                empty_state("No linked vehicle records were found.")
            else:
                show_table(plain(linked.drop(columns="cid")).rename(columns={
                    "vehicle_reference": "Vehicle", "vehicle_type_label": "Vehicle type",
                    "vehicle_manoeuvre_label": "Manoeuvre", "driver_sex_label": "Driver sex",
                    "driver_age_band_label": "Driver age band", "engine_capacity_cc": "Engine (cc)",
                }))
        with v2:
            panel("Casualties involved")
            linked = casualties_all[casualties_all["cid"] == record["cid"]]
            if linked.empty:
                empty_state("No linked casualty records were found.")
            else:
                show_table(plain(linked[["user_type", "casualty_class_label", "age_band_label", "sex_label",
                                         "severity_label"]]).rename(columns={
                    "user_type": "Road user", "casualty_class_label": "Class", "age_band_label": "Age band",
                    "sex_label": "Sex", "severity_label": "Severity",
                }))

    st.download_button(
        "Download this page as CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="uk_collision_records_page.csv",
        mime="text/csv",
    )


# =============================================================================
# 13. ROUTING AND FOOTER
# =============================================================================

if page in DESCRIPTIVE_PAGES and filtered_collisions.empty:
    st.warning("No collision records match the current filters. Adjust or reset the filters in the sidebar.")
elif page == "Overview":
    render_overview()
elif page == "Collision Analytics":
    render_collision_analytics()
elif page == "Road Users":
    render_road_users()
elif page == "Statistical Evidence":
    render_statistical_evidence()
elif page == "Severity Prediction":
    render_prediction()
elif page == "Model Insights":
    render_model_insights()
elif page == "Collision Explorer":
    render_collision_explorer()

st.markdown(
    f'<div class="footer">UK Department for Transport · STATS19 road safety data {year_span(ALL_YEARS)} · '
    f'Analysis and model: {NOTEBOOK_NAME} · Research demonstration</div>',
    unsafe_allow_html=True,
)
