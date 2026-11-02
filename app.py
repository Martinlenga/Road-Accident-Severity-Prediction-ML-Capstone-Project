import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="UK Road Safety Dashboard",
    page_icon="🚦",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "collision": DATA_DIR / "dft-road-casualty-statistics-collision-2025.csv",
    "casualty": DATA_DIR / "dft-road-casualty-statistics-casualty-2025.csv",
    "vehicle": DATA_DIR / "dft-road-casualty-statistics-vehicle-2025.csv",
}

# Simple labels for the most useful coded fields in the source data.
SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
URBAN_RURAL = {1: "Urban", 2: "Rural", 3: "Unallocated"}
SEX = {1: "Male", 2: "Female", 3: "Not applicable", 9: "Unknown", -1: "Unknown"}
CASUALTY_CLASS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
LIGHT = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Unknown",
}
ROAD_TYPE = {
    1: "Roundabout",
    2: "One-way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
}
AGE_BAND = {
    1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20",
    5: "21–25", 6: "26–35", 7: "36–45", 8: "46–55",
    9: "56–65", 10: "66–75", 11: "Over 75", -1: "Unknown",
}

@st.cache_data(show_spinner="Loading the 2025 road safety data...")
def load_data():
    """Read the three related CSV files once and prepare useful fields."""
    missing = [str(path.name) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "These data files are missing from the data folder: " + ", ".join(missing)
        )

    collisions = pd.read_csv(FILES["collision"], low_memory=False)
    casualties = pd.read_csv(FILES["casualty"], low_memory=False)
    vehicles = pd.read_csv(FILES["vehicle"], low_memory=False)

    collisions["date"] = pd.to_datetime(
        collisions["date"], dayfirst=True, errors="coerce"
    )
    collisions["collision_severity_label"] = (
        collisions["collision_severity"].map(SEVERITY).fillna("Unknown")
    )
    collisions["area_label"] = (
        collisions["urban_or_rural_area"].map(URBAN_RURAL).fillna("Unknown")
    )
    collisions["road_type_label"] = (
        collisions["road_type"].map(ROAD_TYPE).fillna("Other / coded value")
    )
    collisions["light_label"] = (
        collisions["light_conditions"].map(LIGHT).fillna("Other / coded value")
    )

    casualties["casualty_severity_label"] = (
        casualties["casualty_severity"].map(SEVERITY).fillna("Unknown")
    )
    casualties["casualty_class_label"] = (
        casualties["casualty_class"].map(CASUALTY_CLASS).fillna("Unknown")
    )
    casualties["age_band_label"] = (
        casualties["age_band_of_casualty"].map(AGE_BAND).fillna("Unknown")
    )
    casualties["sex_label"] = (
        casualties["sex_of_casualty"].map(SEX).fillna("Unknown")
    )

    vehicles["driver_sex_label"] = (
        vehicles["sex_of_driver"].map(SEX).fillna("Unknown")
    )
    vehicles["driver_age_band_label"] = (
        vehicles["age_band_of_driver"].map(AGE_BAND).fillna("Unknown")
    )

    return collisions, casualties, vehicles


st.title("🚦 UK Road Safety Dashboard")
st.caption(
    "Explore reported road collisions, casualties and vehicles in the 2025 dataset. "
    "Use the filters on the left to update every section."
)

try:
    collisions, casualties, vehicles = load_data()
except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.info(
        "Check that all three CSV files are inside the project's data folder "
        "and that their filenames have not changed."
    )
    st.stop()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.title("🔎 Dashboard filters")
st.sidebar.caption("Choose filters below. The dashboard updates automatically.")

valid_dates = collisions["date"].dropna()
if valid_dates.empty:
    st.error("The collision date column could not be read.")
    st.stop()

min_date = valid_dates.min().date()
max_date = valid_dates.max().date()

date_range = st.sidebar.date_input(
    "Accident date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

severity_options = ["Fatal", "Serious", "Slight"]
selected_severity = st.sidebar.multiselect(
    "Collision severity",
    options=severity_options,
    default=severity_options,
)

area_options = sorted(collisions["area_label"].dropna().unique().tolist())
selected_areas = st.sidebar.multiselect(
    "Area type",
    options=area_options,
    default=area_options,
)

speed_options = sorted(
    int(x) for x in collisions["speed_limit"].dropna().unique()
    if pd.notna(x) and int(x) > 0
)
selected_speeds = st.sidebar.multiselect(
    "Speed limit (mph)",
    options=speed_options,
    default=speed_options,
)

road_options = sorted(collisions["road_type_label"].dropna().unique().tolist())
selected_roads = st.sidebar.multiselect(
    "Road type",
    options=road_options,
    default=road_options,
)

# Apply filters to the collision table.
filtered = collisions[
    collisions["date"].dt.date.between(start_date, end_date)
    & collisions["collision_severity_label"].isin(selected_severity)
    & collisions["area_label"].isin(selected_areas)
    & collisions["speed_limit"].isin(selected_speeds)
    & collisions["road_type_label"].isin(selected_roads)
].copy()

# Find matching rows in the related tables without joining and multiplying rows.
collision_ids = set(filtered["collision_index"].dropna().astype(str))
casualty_match = casualties[
    casualties["collision_index"].astype(str).isin(collision_ids)
].copy()
vehicle_match = vehicles[
    vehicles["collision_index"].astype(str).isin(collision_ids)
].copy()

st.sidebar.divider()
st.sidebar.write(f"**Collisions matching filters:** {len(filtered):,}")
if st.sidebar.button("Reset filters"):
    st.rerun()

# ---------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------
st.subheader("At a glance")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Collisions", f"{len(filtered):,}")
m2.metric("Casualties", f"{len(casualty_match):,}")
m3.metric("Vehicles involved", f"{len(vehicle_match):,}")
fatal_count = int((filtered["collision_severity_label"] == "Fatal").sum())
m4.metric("Fatal collisions", f"{fatal_count:,}")

if filtered.empty:
    st.warning("No records match these filters. Try widening the date range or selecting more options.")
    st.stop()

tab_overview, tab_collisions, tab_people, tab_vehicles, tab_data = st.tabs(
    ["📊 Overview", "🚧 Collision patterns", "🧍 Casualties", "🚗 Vehicles", "📁 Data"]
)

with tab_overview:
    left, right = st.columns(2)

    with left:
        st.subheader("Collisions by severity")
        severity_counts = (
            filtered["collision_severity_label"]
            .value_counts()
            .reindex(["Fatal", "Serious", "Slight"], fill_value=0)
        )
        st.bar_chart(severity_counts)

    with right:
        st.subheader("Collisions by area")
        area_counts = filtered["area_label"].value_counts()
        st.bar_chart(area_counts)

    st.subheader("Collisions over time")
    by_month = (
        filtered.assign(month=filtered["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .size()
        .rename("Collisions")
    )
    st.line_chart(by_month)

    st.subheader("Where collisions happened")
    map_data = filtered.dropna(subset=["latitude", "longitude"])[
        ["latitude", "longitude"]
    ].copy()
    map_data = map_data[
        map_data["latitude"].between(49, 61)
        & map_data["longitude"].between(-9, 3)
    ]
    if not map_data.empty:
        if len(map_data) > 5000:
            map_data = map_data.sample(5000, random_state=42)
            st.caption("Map displays a sample of up to 5,000 matching collision locations.")
        st.map(map_data, latitude="latitude", longitude="longitude")
    else:
        st.info("No valid map coordinates are available for the selected filters.")

with tab_collisions:
    st.subheader("Collision patterns")
    c1, c2 = st.columns(2)

    with c1:
        st.write("**Collisions by speed limit (mph)**")
        speed_counts = (
            filtered[filtered["speed_limit"] > 0]["speed_limit"]
            .value_counts()
            .sort_index()
        )
        st.bar_chart(speed_counts)

    with c2:
        st.write("**Collisions by road type**")
        st.bar_chart(filtered["road_type_label"].value_counts())

    st.write("**Collisions by light condition**")
    st.bar_chart(filtered["light_label"].value_counts())

    st.write("**Most common local authority codes**")
    if "local_authority_district" in filtered.columns:
        authority_counts = (
            filtered["local_authority_district"]
            .replace(-1, pd.NA)
            .dropna()
            .astype(str)
            .value_counts()
            .head(10)
        )
        st.bar_chart(authority_counts)
        st.caption("Local authority values are shown as source dataset codes.")

with tab_people:
    st.subheader("People involved in collisions")
    p1, p2 = st.columns(2)

    with p1:
        st.write("**Casualties by severity**")
        st.bar_chart(casualty_match["casualty_severity_label"].value_counts())

    with p2:
        st.write("**Casualties by age band**")
        age_order = list(AGE_BAND.values())
        age_counts = casualty_match["age_band_label"].value_counts()
        age_counts = age_counts.reindex(
            [a for a in age_order if a in age_counts.index], fill_value=0
        )
        st.bar_chart(age_counts)

    st.write("**Casualties by road-user class**")
    st.bar_chart(casualty_match["casualty_class_label"].value_counts())

    st.write("**Casualties by sex**")
    st.bar_chart(casualty_match["sex_label"].value_counts())

with tab_vehicles:
    st.subheader("Vehicles and drivers")
    v1, v2 = st.columns(2)

    with v1:
        st.write("**Driver age bands**")
        st.bar_chart(vehicle_match["driver_age_band_label"].value_counts())

    with v2:
        st.write("**Driver sex (source categories)**")
        st.bar_chart(vehicle_match["driver_sex_label"].value_counts())

    if "vehicle_type" in vehicle_match.columns:
        st.write("**Vehicle type codes**")
        st.bar_chart(vehicle_match["vehicle_type"].value_counts().head(15))
        st.caption("Vehicle type is displayed using the original dataset codes.")

with tab_data:
    st.subheader("Explore the filtered collision records")
    st.write("Use the table search, sort and download controls to explore the records.")
    display_columns = [
        c for c in [
            "date", "time", "collision_severity_label", "number_of_vehicles",
            "number_of_casualties", "speed_limit", "road_type_label",
            "area_label", "latitude", "longitude", "local_authority_district",
        ] if c in filtered.columns
    ]
    st.dataframe(
        filtered[display_columns].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered collision data (CSV)",
        data=csv_bytes,
        file_name="filtered_road_collisions_2025.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Source: UK Department for Transport road casualty statistics, 2025. "
    "Counts reflect records in the supplied files and the filters selected. "
    "Some fields in the source are numeric codes; where not translated here, "
    "the original code is shown. This dashboard is for exploration, not an official "
    "safety assessment."
)
