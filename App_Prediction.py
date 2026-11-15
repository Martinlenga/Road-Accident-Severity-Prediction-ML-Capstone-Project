import inspect
import base64
import json
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

st.set_page_config(page_title="UK Road Safety Intelligence | Dashboard & Severity Prediction", page_icon="🚦", layout="wide")
# V6 is responsive for phone browsers. Public sharing requires deployment to a hosted URL; localhost is local-only.

# Feature checks so the app also runs on older Streamlit
_ALT = inspect.signature(st.altair_chart).parameters
_DF = inspect.signature(st.dataframe).parameters
CLICK = "on_select" in _ALT
ROWSEL = "on_select" in _DF
rerun = getattr(st, "rerun", None) or st.experimental_rerun


try:
    _V = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _V = (1, 50)


def _wide(params):
    # Streamlit 1.50+ uses width="stretch"; older versions use use_container_width
    if _V >= (1, 50) and "width" in params:
        return {"width": "stretch"}
    return {"use_container_width": True}


def show(chart, key=None):
    if key and CLICK:
        st.altair_chart(chart, key=key, on_select="rerun", **_wide(_ALT))
    else:
        st.altair_chart(chart, **_wide(_ALT))


@contextmanager
def card(title):
    try:
        box = st.container(border=True)
    except TypeError:
        box = st.container()
    with box:
        st.markdown(f'<div class="ct">{title}</div>', unsafe_allow_html=True)
        yield


# Data location: finds the CSVs by keyword, so filenames can differ
APP_DIR = Path(__file__).parent
SEARCH_DIRS = [APP_DIR / "data", APP_DIR, APP_DIR.parent / "data"]


def find_file(keyword):
    for d in SEARCH_DIRS:
        if d.exists():
            hits = sorted(p for p in d.glob("*.csv") if keyword in p.name.lower())
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"No CSV with '{keyword}' in its name found in: " + ", ".join(str(d) for d in SEARCH_DIRS)
    )


# Palette
INK, BLUE, TEAL = "#12263F", "#1D4E89", "#2A9D8F"
AMBER, RED, GREEN, GREY = "#F2A900", "#C8102E", "#2E9E6B", "#8A97A8"
SLATE = "#4A6FA5"

SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLORS = {"Fatal": RED, "Serious": AMBER, "Slight": GREEN}
SEVERITY_RGB = {"Fatal": [200, 16, 46, 230], "Serious": [242, 169, 0, 200], "Slight": [46, 158, 107, 150]}
AREA_COLORS = {"Urban": BLUE, "Rural": TEAL, "Unallocated": GREY, "Unknown": GREY}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOURS = [f"{h:02d}:00" for h in range(24)]

# Lookups
SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}
URBAN_RURAL = {1: "Urban", 2: "Rural", 3: "Unallocated"}
SEX = {1: "Male", 2: "Female", 3: "Not applicable", 9: "Unknown", -1: "Unknown"}
CASUALTY_CLASS = {1: "Driver or rider", 2: "Passenger", 3: "Pedestrian"}
LIGHT = {
    1: "Daylight", 4: "Darkness - lights lit", 5: "Darkness - lights unlit",
    6: "Darkness - no lighting", 7: "Darkness - lighting unknown", -1: "Unknown",
}
LIGHT_ORDER = list(LIGHT.values())[:-1] + ["Unknown"]
ROAD_TYPE = {
    1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
    6: "Single carriageway", 7: "Slip road", 9: "Unknown",
}
AGE_BAND = {
    1: "0–5", 2: "6–10", 3: "11–15", 4: "16–20", 5: "21–25", 6: "26–35",
    7: "36–45", 8: "46–55", 9: "56–65", 10: "66–75", 11: "Over 75", -1: "Unknown",
}
AGE_ORDER = list(AGE_BAND.values())[:-1] + ["Unknown"]
VEHICLE_TYPE = {
    1: "Pedal cycle", 2: "Motorcycle ≤50cc", 3: "Motorcycle ≤125cc",
    4: "Motorcycle 125–500cc", 5: "Motorcycle >500cc", 8: "Taxi / private hire",
    9: "Car", 10: "Minibus", 11: "Bus or coach", 16: "Ridden horse",
    17: "Agricultural vehicle", 18: "Tram", 19: "Van / goods ≤3.5t",
    20: "Goods 3.5–7.5t", 21: "Goods >7.5t", 22: "Mobility scooter",
    23: "Electric motorcycle", 90: "Other vehicle", 97: "Motorcycle (cc unknown)",
    98: "Goods (weight unknown)", 99: "Unknown vehicle", -1: "Unknown",
}
USER_GROUP = {0: "Pedestrian", 1: "Cyclist", 8: "Car occupant", 9: "Car occupant",
              10: "Bus / coach", 11: "Bus / coach",
              19: "Van / goods", 20: "Van / goods", 21: "Van / goods", 98: "Van / goods"}
USER_GROUP.update({c: "Motorcyclist" for c in (2, 3, 4, 5, 23, 97, 103, 104, 105, 106)})
VULNERABLE = ["Pedestrian", "Cyclist", "Motorcyclist"]

COLLISION_COLS = [
    "collision_index", "date", "time", "collision_severity", "urban_or_rural_area",
    "speed_limit", "road_type", "light_conditions", "latitude", "longitude",
    "number_of_vehicles", "number_of_casualties",
]
CASUALTY_COLS = ["collision_index", "casualty_severity", "casualty_class",
                 "age_band_of_casualty", "sex_of_casualty", "casualty_type"]
VEHICLE_COLS = ["collision_index", "sex_of_driver", "age_band_of_driver", "vehicle_type"]


def read_columns(path, wanted):
    available = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, usecols=[c for c in wanted if c in available], low_memory=False)


@st.cache_resource(show_spinner="Loading road safety data...")
def load_data():
    c = read_columns(find_file("collision"), COLLISION_COLS)
    ca = read_columns(find_file("casualt"), CASUALTY_COLS)
    v = read_columns(find_file("vehicle"), VEHICLE_COLS)
    for df in (c, ca, v):
        df["collision_index"] = df["collision_index"].astype(str)

    c["date"] = pd.to_datetime(c["date"], dayfirst=True, errors="coerce")
    c["collision_severity_label"] = c["collision_severity"].map(SEVERITY).fillna("Unknown")
    c["area_label"] = c["urban_or_rural_area"].map(URBAN_RURAL).fillna("Unknown")
    c["road_type_label"] = c["road_type"].map(ROAD_TYPE).fillna("Other / coded value")
    c["light_label"] = c["light_conditions"].map(LIGHT).fillna("Other / coded value")
    sl = c["speed_limit"]
    c["speed_label"] = np.where(sl > 0, sl.fillna(0).astype(int).astype(str), "Unknown")
    c["weekday"] = c["date"].dt.day_name().fillna("Unknown")
    c["month_label"] = c["date"].dt.strftime("%Y-%m").fillna("Unknown")
    hour = pd.to_numeric(c["time"].astype(str).str[:2], errors="coerce")
    idx = hour.where(hour.between(0, 23), 24).fillna(24).astype(int).to_numpy()
    c["hour_label"] = np.array(HOURS + ["Unknown"])[idx]
    c["is_fatal"] = c["collision_severity"] == 1
    c["is_dark"] = c["light_conditions"].isin([4, 5, 6, 7])

    ca["casualty_severity_label"] = ca["casualty_severity"].map(SEVERITY).fillna("Unknown")
    ca["age_band_label"] = ca["age_band_of_casualty"].map(AGE_BAND).fillna("Unknown")
    ca["sex_label"] = ca["sex_of_casualty"].map(SEX).fillna("Unknown")
    if "casualty_type" in ca.columns:
        ca["user_type"] = ca["casualty_type"].map(USER_GROUP).fillna("Other")
    else:
        ca["user_type"] = ca["casualty_class"].map({3: "Pedestrian"}).fillna("Other")
    ca["is_ksi"] = ca["casualty_severity"].isin([1, 2])
    ca["is_vru"] = ca["user_type"].isin(VULNERABLE)

    v["driver_sex_label"] = v["sex_of_driver"].map(SEX).fillna("Unknown")
    v["driver_age_band_label"] = v["age_band_of_driver"].map(AGE_BAND).fillna("Unknown")
    if "vehicle_type" in v.columns:
        v["vehicle_type_label"] = v["vehicle_type"].map(VEHICLE_TYPE).fillna(
            v["vehicle_type"].apply(lambda x: f"Code {x}"))

    for df, cols in (
        (c, ["collision_severity_label", "area_label", "road_type_label", "light_label",
             "speed_label", "weekday", "month_label", "hour_label"]),
        (ca, ["casualty_severity_label", "age_band_label", "sex_label"]),
    ):
        for col in cols:
            df[col] = df[col].astype("category")
    return c, ca, v


def metrics(coll, cas):
    n, nc = len(coll), len(cas)
    return dict(
        n=n, nc=nc,
        killed=int((cas["casualty_severity"] == 1).sum()),
        serious=int((cas["casualty_severity"] == 2).sum()),
        vru_pct=cas["is_vru"].mean() * 100 if nc else 0,
        dark_pct=coll["is_dark"].mean() * 100 if n else 0,
    )


@st.cache_resource
def baseline():
    c, ca, _ = load_data()
    return metrics(c, ca)


def _dashboard_watermark_css():
    """Optional background asset. The app remains fully functional if it is absent."""
    image_path = APP_DIR / "road_safety_watermark.png"
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"""
    .stApp {{
        background-image:
            linear-gradient(rgba(237,241,246,.86), rgba(237,241,246,.86)),
            url('data:image/png;base64,{encoded}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    section[data-testid="stSidebar"] > div {{
        background: rgba(255,255,255,.91);
        backdrop-filter: blur(2px);
    }}
    """


_WATERMARK_CSS = _dashboard_watermark_css()

# Styling: light page, white cards
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Barlow', sans-serif; }}
    span[data-testid="stIconMaterial"], [class*="material-symbols"] {{
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }}
    .stApp {{ background: #EDF1F6; }}
    {_WATERMARK_CSS}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px; }}
    h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif !important; color: {INK}; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #fff; border: none; border-radius: 12px; box-shadow: 0 1px 4px rgba(18,38,63,.12); }}
    .ct {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.15rem;
           color: {INK}; margin-bottom: .2rem; }}
    .topbar {{ background: linear-gradient(120deg, {INK} 0%, {BLUE} 100%); border-radius: 12px;
               padding: .7rem 1.2rem; margin-bottom: .8rem; display: flex;
               justify-content: space-between; align-items: center; position: relative; overflow: hidden; }}
    .topbar::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 4px;
        background: repeating-linear-gradient(90deg, {AMBER} 0 30px, transparent 30px 50px); }}
    .topbar .t {{ color: #fff; font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.8rem; }}
    .topbar .p {{ color: #D5E1F2; font-size: 1rem; font-weight: 500; }}
    .kpi {{ background: #fff; border-radius: 12px; padding: 0 .8rem; height: 100px; box-sizing: border-box;
            display: flex; flex-direction: column; justify-content: center; overflow: hidden;
            border-left: 6px solid var(--c); box-shadow: 0 1px 4px rgba(18,38,63,.12); }}
    .kpi .label, .kpi .sub {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi .label {{ color: #5B6B80; font-size: .8rem; font-weight: 600; }}
    .kpi .value {{ color: var(--c); font-family: 'Barlow Condensed', sans-serif; white-space: nowrap;
                   font-size: clamp(1.6rem, 2.1vw, 2.1rem); font-weight: 700; line-height: 1.1; }}
    .kpi .sub {{ color: #5B6B80; font-size: .75rem; min-height: 1rem; }}
    .fl {{ color: #5B6B80; font-size: .85rem; font-weight: 600; margin: .6rem 0 .2rem; }}
    section[data-testid="stSidebar"] div[data-testid="stPopover"] button {{ background: #fff; }}
    .up {{ color: {RED}; font-weight: 600; }} .down {{ color: {GREEN}; font-weight: 600; }}
    .flat {{ color: #7A8799; }}
    .chip {{ display: inline-block; background: {BLUE}; color: #fff; border-radius: 999px;
             padding: .1rem .7rem; margin: 0 .3rem .3rem 0; font-size: .85rem; }}
    .legend span {{ margin-right: .9rem; font-size: .85rem; color: #44546A; }}
    section[data-testid="stSidebar"] {{ border-right: 4px solid {AMBER}; }}
    button[data-baseweb="tab"] p {{ font-weight: 600; font-size: 1rem; }}

    /* V6 mobile responsiveness. Desktop geometry is unchanged. */
    @media (max-width: 768px) {{
        .block-container {{ padding: .55rem .55rem 1rem; max-width: 100%; }}
        .topbar {{ padding: .65rem .75rem; border-radius: 10px; align-items: flex-start; gap: .4rem; }}
        .topbar .t {{ font-size: 1.35rem; line-height: 1.05; }}
        .topbar .t span {{ display: block; margin-left: 0 !important; margin-top: .18rem; font-size: .8rem !important; }}
        .topbar .p {{ font-size: .8rem; white-space: nowrap; }}
        .kpi {{ height: 92px; padding: 0 .7rem; }}
        .kpi .label {{ font-size: .72rem; }}
        .kpi .value {{ font-size: 1.75rem; }}
        .kpi .sub {{ font-size: .68rem; }}
        .ct {{ font-size: 1rem; }}
        .chip {{ font-size: .75rem; }}
        section[data-testid="stSidebar"] {{ min-width: 82vw !important; max-width: 82vw !important; }}
        div[data-testid="stHorizontalBlock"] {{ gap: .55rem; }}
        button[data-baseweb="tab"] p {{ font-size: .85rem; }}
        div[data-testid="stDataFrame"] {{ overflow-x: auto; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
try:
    collisions, casualties, vehicles = load_data()
except Exception as exc:
    st.error(f"Could not load the data: {exc}")
    st.info("Put the three CSV files in a `data` folder next to app.py. Their names must contain "
            "'collision', 'casualt' and 'vehicle'.")
    st.stop()

valid_dates = collisions["date"].dropna()
if valid_dates.empty:
    st.error("The collision date column could not be read.")
    st.stop()
min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
period = str(min_date.year) if min_date.year == max_date.year else f"{min_date.year}–{max_date.year}"

st.markdown(f'<div class="topbar"><span class="t">🚦 UK Road Safety Dashboard <span style="font-size:1rem;font-family:Barlow,sans-serif;font-weight:500;color:#D5E1F2;margin-left:.55rem">& Severity Prediction</span></span>'
            f'<span class="p">{period}</span></div>', unsafe_allow_html=True)

# Sidebar filters
severity_options = SEVERITY_ORDER
area_options = sorted(collisions["area_label"].dropna().unique().tolist())
speed_options = sorted((s for s in collisions["speed_label"].unique() if s != "Unknown"), key=int)
if (collisions["speed_label"] == "Unknown").any():
    speed_options.append("Unknown")
road_options = sorted(collisions["road_type_label"].dropna().unique().tolist())

FILTERS = {"severity": severity_options, "area": area_options,
           "speed": speed_options, "road": road_options}
for name, opts in FILTERS.items():
    st.session_state.setdefault(f"{name}_all", True)
    for i in range(len(opts)):
        st.session_state.setdefault(f"{name}_{i}", True)
st.session_state.setdefault("ver", 0)


def reset_filters():
    for name, opts in FILTERS.items():
        st.session_state[f"{name}_all"] = True
        for i in range(len(opts)):
            st.session_state[f"{name}_{i}"] = True
    st.session_state["f_dates"] = (min_date, max_date)
    st.session_state["ver"] += 1


def clear_drill():
    st.session_state["ver"] += 1


POPOVER = hasattr(st, "popover")
_POP = inspect.signature(st.popover).parameters if POPOVER else {}


def dropdown(label, name, options):
    """Power BI style slicer: a dropdown with 'Select all' and a tick box per option."""
    n, all_key = len(options), f"{name}_all"

    def pick_all():
        for i in range(n):
            st.session_state[f"{name}_{i}"] = st.session_state[all_key]

    def pick_one():
        st.session_state[all_key] = all(st.session_state[f"{name}_{i}"] for i in range(n))

    chosen = [o for i, o in enumerate(options) if st.session_state[f"{name}_{i}"]]
    if len(chosen) == n:
        text = "All"
    elif not chosen:
        text = "None"
    elif len(chosen) <= 2:
        text = ", ".join(chosen)
    else:
        text = f"{len(chosen)} selected"

    st.sidebar.markdown(f'<div class="fl">{label}</div>', unsafe_allow_html=True)
    box = st.sidebar.popover(text, **_wide(_POP)) if POPOVER else st.sidebar.expander(text)
    with box:
        st.checkbox("Select all", key=all_key, on_change=pick_all)
        for i, o in enumerate(options):
            st.checkbox(str(o), key=f"{name}_{i}", on_change=pick_one)
    return chosen


def range_text(v):
    v = tuple(v) if isinstance(v, (tuple, list)) else (v,)
    end = f"{v[1]:%d %b %Y}" if len(v) == 2 else "…"
    return f"{v[0]:%d %b %Y} – {end}"


st.sidebar.markdown('<div class="fl">Date range</div>', unsafe_allow_html=True)
st.session_state.setdefault("f_dates", (min_date, max_date))
_label = range_text(st.session_state["f_dates"])
box = st.sidebar.popover(_label, **_wide(_POP)) if POPOVER else st.sidebar.expander(_label)
with box:
    date_range = st.date_input("Date range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date,
                               key="f_dates", label_visibility="collapsed")
sev_sel = dropdown("Severity", "severity", severity_options)
area_sel = dropdown("Area", "area", area_options)
speed_sel = dropdown("Speed limit (mph)", "speed", speed_options)
road_sel = dropdown("Road type", "road", road_options)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

base = collisions[
    (collisions["date"] >= pd.Timestamp(start_date))
    & (collisions["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
    & collisions["collision_severity_label"].isin(sev_sel)
    & collisions["area_label"].isin(area_sel)
    & collisions["speed_label"].isin(speed_sel)
    & collisions["road_type_label"].isin(road_sel)
]
st.sidebar.divider()
st.sidebar.write(f"**{len(base):,}** collisions match")
st.sidebar.button("Reset filters", on_click=reset_filters, type="primary")
if not CLICK:
    st.sidebar.info("For click-to-filter charts, run:\n\n`python -m pip install -U streamlit altair`")

if base.empty:
    st.warning("No records match these filters.")
    st.stop()

# Drill-through state (clicks on charts)
DIMS = {"severity": "collision_severity_label", "area": "area_label", "road": "road_type_label",
        "speed": "speed_label", "light": "light_label", "month": "month_label",
        "weekday": "weekday", "hour": "hour_label"}
DRILL_NAMES = {"severity": "Severity", "area": "Area", "road": "Road type", "speed": "Speed limit",
               "light": "Light", "month": "Month", "weekday": "Day", "hour": "Hour", "user": "Road user"}


def ckey(dim):
    return f"drill_{dim}_{st.session_state['ver']}"


def get_sel(dim):
    if not CLICK:
        return []
    try:
        pts = st.session_state.get(ckey(dim))["selection"]["pick"]
    except Exception:
        return []
    return [str(p["label"]) for p in pts if "label" in p]


sel = {d: get_sel(d) for d in list(DIMS) + ["user"]}


@st.cache_data
def user_ids(vals):
    return set(casualties.loc[casualties["user_type"].isin(vals), "collision_index"])


def apply_drill(df, skip=None):
    for dim, col in DIMS.items():
        if dim != skip and sel[dim]:
            df = df[df[col].isin(sel[dim])]
    if skip != "user" and sel["user"]:
        df = df[df["collision_index"].isin(user_ids(tuple(sel["user"])))]
    return df


def cas_for(view):
    return casualties[casualties["collision_index"].isin(set(view["collision_index"]))]


def view(dim):
    """Data for a chart, ignoring that chart's own click."""
    return apply_drill(base, skip=dim)


cur = apply_drill(base)
if cur.empty:
    st.session_state["ver"] += 1
    rerun()
cas_cur = cas_for(cur)

active = {d: v for d, v in sel.items() if v}
if active:
    c1, c2 = st.columns([6, 1])
    chips = "".join(f'<span class="chip"><b>{DRILL_NAMES[d]}:</b> {", ".join(v)}</span>' for d, v in active.items())
    c1.markdown(chips, unsafe_allow_html=True)
    c2.button("Clear", on_click=clear_drill)
elif CLICK:
    st.caption("💡 Click any chart to filter the whole dashboard")

# KPI cards (compared with the whole dataset)
m, b = metrics(cur, cas_cur), baseline()
days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)


def delta(val, ref):
    d = val - ref
    if abs(d) < 0.05:
        return ""
    cls, arrow = ("up", "▲") if d > 0 else ("down", "▼")
    return f'<span class="{cls}">{arrow}{abs(d):.1f}</span>'


def share(x):
    return f"{x / m['nc'] * 100:.1f}% of casualties" if m["nc"] else "–"


def kpi(col, label, value, sub, color, tip=""):
    col.markdown(f'<div class="kpi" title="{tip}" style="--c:{color}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


VS_ALL = "The arrow shows the change in percentage points compared with the whole dataset."
k = st.columns(6, gap="small")
kpi(k[0], "Collisions", f"{m['n']:,}", f"{m['n'] / days:,.0f} a day", BLUE)
kpi(k[1], "Casualties", f"{m['nc']:,}", f"{m['nc'] / m['n']:.1f} per collision", SLATE)
kpi(k[2], "Killed", f"{m['killed']:,}", share(m["killed"]), RED)
kpi(k[3], "Seriously injured", f"{m['serious']:,}", share(m["serious"]), AMBER)
kpi(k[4], "Vulnerable users", f"{m['vru_pct']:.0f}%", "of casualties " + delta(m["vru_pct"], b["vru_pct"]), TEAL,
    "Pedestrians, cyclists and motorcyclists as a share of casualties. " + VS_ALL)
kpi(k[5], "Dark collisions", f"{m['dark_pct']:.0f}%", "of collisions " + delta(m["dark_pct"], b["dark_pct"]), INK,
    "Share of collisions in darkness. " + VS_ALL)
st.write("")


# Chart helpers
def _opacity(dim):
    if sel[dim]:
        return alt.condition(alt.FieldOneOfPredicate(field="label", oneOf=sel[dim]),
                             alt.value(1), alt.value(0.3))
    return alt.value(1)


def drill_bar(df, dim, color=BLUE, colors=None, order=None, horizontal=False, height=230,
              vtitle="Collisions", vfmt=",", scale_color=False, extra=None):
    df = df.copy()
    df["label"] = df["label"].astype(str)
    pick = alt.selection_point(name="pick", fields=["label"])
    if horizontal:
        x, y = alt.X("value:Q", title=None), alt.Y("label:N", sort=order or "-x", title=None)
    else:
        x = alt.X("label:N", sort=order or "-y", title=None,
                  axis=alt.Axis(labelAngle=-45 if len(df) > 8 else 0, labelOverlap=True))
        y = alt.Y("value:Q", title=None)
    if colors:
        col = alt.Color("label:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None)
    elif scale_color:
        col = alt.Color("value:Q", scale=alt.Scale(range=[AMBER, RED]), legend=None)
    else:
        col = alt.value(color)
    tips = [alt.Tooltip("label:N", title="Category"), alt.Tooltip("value:Q", title=vtitle, format=vfmt)]
    tips += [alt.Tooltip(f"{c}:Q", title=t, format=",") for c, t in (extra or {}).items()]
    chart = (alt.Chart(df).mark_bar(cornerRadius=3)
             .encode(x=x, y=y, color=col, opacity=_opacity(dim), tooltip=tips)
             .add_params(pick).properties(height=height))
    show(chart, ckey(dim))


def drill_donut(df, dim, colors):
    df = df.copy()
    df["label"] = df["label"].astype(str)
    present = [l for l in colors if l in set(df["label"])]
    pick = alt.selection_point(name="pick", fields=["label"])
    arc = (alt.Chart(df).mark_arc(innerRadius=50, outerRadius=85, stroke="white")
           .encode(theta=alt.Theta("value:Q"),
                   color=alt.Color("label:N", scale=alt.Scale(domain=present, range=[colors[l] for l in present]),
                                   legend=alt.Legend(orient="right", title=None)),
                   opacity=_opacity(dim),
                   tooltip=[alt.Tooltip("label:N", title="Category"), alt.Tooltip("value:Q", title="Collisions", format=",")])
           .add_params(pick))
    total = (alt.Chart(pd.DataFrame({"t": [f"{int(df['value'].sum()):,}"]}))
             .mark_text(size=18, fontWeight="bold", color=INK).encode(text="t:N"))
    show(alt.layer(arc, total).properties(height=190), ckey(dim))


def drill_stack(df, dim, order, legend=False, label_expr=None, height=250):
    df = df.copy()
    df["label"] = df["label"].astype(str)
    pick = alt.selection_point(name="pick", fields=["label"])
    axis = alt.Axis(labelExpr=label_expr, labelLimit=150) if label_expr else alt.Axis(labelLimit=150)
    chart = (alt.Chart(df).mark_bar()
             .encode(x=alt.X("n:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
                     y=alt.Y("label:N", sort=order, title=None, axis=axis),
                     color=alt.Color("severity:N", sort=SEVERITY_ORDER,
                                     scale=alt.Scale(domain=SEVERITY_ORDER,
                                                     range=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER]),
                                     legend=alt.Legend(orient="bottom", title=None) if legend else None),
                     opacity=_opacity(dim),
                     tooltip=["label:N", "severity:N", alt.Tooltip("n:Q", title="Collisions", format=",")])
             .add_params(pick).properties(height=height))
    show(chart, ckey(dim))


def counts(series):
    return series.value_counts().rename_axis("label").reset_index(name="value")


def speed_rates(df):
    g = (df[df["speed_label"] != "Unknown"].groupby("speed_label", observed=True)
         .agg(n=("is_fatal", "size"), fatal=("is_fatal", "sum")).reset_index())
    g = g[g["n"] > 0].copy()
    g["label"] = g["speed_label"].astype(str)
    g["value"] = g["fatal"] / g["n"] * 1000
    return g.assign(o=g["label"].astype(int)).sort_values("o").drop(columns=["o", "speed_label"])


def draw_map(df):
    d = df.dropna(subset=["latitude", "longitude"])[["latitude", "longitude", "collision_severity_label", "date"]]
    d = d[d["latitude"].between(49, 61) & d["longitude"].between(-9, 3)].copy()
    if d.empty:
        st.info("No map coordinates for this selection.")
        return
    d["severity"] = d["collision_severity_label"].astype(str)
    fatal, rest = d[d["severity"] == "Fatal"], d[d["severity"] != "Fatal"]
    if len(rest) > 7000:
        rest = rest.sample(7000, random_state=42)
    d = pd.concat([rest, fatal])  # fatal drawn last, on top
    d["when"] = d["date"].dt.strftime("%d %b %Y")
    d["color"] = d["severity"].map(SEVERITY_RGB)
    d = d[["latitude", "longitude", "severity", "when", "color"]]
    try:
        import pydeck as pdk
        layer = pdk.Layer("ScatterplotLayer", d, get_position="[longitude, latitude]",
                          get_fill_color="color", get_radius=600, radius_min_pixels=2,
                          radius_max_pixels=7, pickable=True)
        deck = pdk.Deck(layers=[layer], map_style=None,
                        initial_view_state=pdk.ViewState(latitude=54.2, longitude=-2.6, zoom=4.6),
                        tooltip={"text": "{severity} · {when}"})
        st.pydeck_chart(deck)
    except Exception:
        d2 = d.assign(color=d["severity"].map(SEVERITY_COLORS))
        try:
            st.map(d2, latitude="latitude", longitude="longitude", color="color", size=60)
        except TypeError:
            st.map(d2[["latitude", "longitude"]])
    st.markdown(f'<div class="legend"><span style="color:{RED}">●</span> Fatal'
                f'<span style="color:{AMBER}">●</span> Serious <span style="color:{GREEN}">●</span> Slight'
                f' <span>· map shows up to 7,000 locations, all fatal ones included</span></div>',
                unsafe_allow_html=True)



# ---------------------------------------------------------
# Prediction model compatibility and contract-driven module
# ---------------------------------------------------------
class CategoricalStringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.categorical_columns_ = [
            column for column in self.categorical_columns if column in X.columns
        ]
        return self

    def transform(self, X):
        transformed = X.copy()
        columns = getattr(self, "categorical_columns_", self.categorical_columns)
        for column in columns:
            if column in transformed.columns:
                values = transformed[column].astype(object)
                transformed[column] = values.where(values.notna(), "Missing").map(str)
        return transformed


class CatBoostClassifierAdapter(ClassifierMixin, BaseEstimator):
    def __init__(self, categorical_columns, iterations=350, depth=6,
                 learning_rate=0.08, loss_function="MultiClass",
                 eval_metric="TotalF1:average=Macro",
                 auto_class_weights="Balanced", random_seed=42,
                 thread_count=-1, verbose=False, allow_writing_files=False):
        self.categorical_columns = categorical_columns
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.auto_class_weights = auto_class_weights
        self.random_seed = random_seed
        self.thread_count = thread_count
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files

    def fit(self, X, y, sample_weight=None):
        from catboost import CatBoostClassifier
        cat_features = [column for column in self.categorical_columns if column in X.columns]
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, loss_function=self.loss_function,
            eval_metric=self.eval_metric, cat_features=cat_features,
            auto_class_weights=self.auto_class_weights,
            random_seed=self.random_seed, thread_count=self.thread_count,
            verbose=self.verbose, allow_writing_files=self.allow_writing_files,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X)).reshape(-1)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)


class DecisionAdjustedModel:
    def __init__(self, fitted_estimator, class_multipliers):
        self.fitted_estimator = fitted_estimator
        self.class_multipliers = dict(class_multipliers)
        self.classes_ = np.asarray(fitted_estimator.classes_)

    def predict_proba(self, X):
        return self.fitted_estimator.predict_proba(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        multipliers = np.asarray(
            [self.class_multipliers.get(label, 1.0) for label in self.classes_],
            dtype=float,
        )
        return self.classes_[np.argmax(probabilities * multipliers, axis=1)]


MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "stats19_severity_model.joblib"
CONTRACT_PATH = MODEL_DIR / "model_contract.json"
KEY_PREDICTORS = [
    "driver_age_mean", "vulnerable_road_user_count",
    "vulnerable_road_user_involved", "driver_age_min",
    "vehicle_type_first", "trunk_road_flag", "engine_cc_mean",
    "road_type", "time_band", "speed_limit",
]
PREDICTOR_LABELS = {
    "driver_age_mean": "Mean driver age",
    "vulnerable_road_user_count": "Vulnerable road-user count",
    "vulnerable_road_user_involved": "Vulnerable road user involved",
    "driver_age_min": "Youngest driver age",
    "vehicle_type_first": "First vehicle type",
    "trunk_road_flag": "Trunk road flag",
    "engine_cc_mean": "Mean engine capacity (cc)",
    "road_type": "Road type", "time_band": "Time band",
    "speed_limit": "Speed limit (mph)",
}

# Human-readable labels shown in the UI; values remain the exact STATS19/model codes.
VEHICLE_TYPE_INPUT = {label: code for code, label in VEHICLE_TYPE.items()}
ROAD_TYPE_INPUT = {
    "Roundabout": 1,
    "One-way street": 2,
    "Dual carriageway": 3,
    "Single carriageway": 6,
    "Slip road": 7,
    "Unknown": 9,
    "Missing / not recorded": -1,
}
BINARY_INPUT = {"No": 0, "Yes": 1}

# Decoders for the remaining categorical model predictors.
# Mappings already validated in the project notebook are explicit; common
# STATS19 fields follow the DfT coding used by the same five-year extract.
ADVANCED_CODE_LABELS = {
    "junction_detail": {
        0:"Not at or within 20 metres of a junction",1:"Roundabout",2:"Mini-roundabout",
        3:"T or staggered junction",5:"Slip road",6:"Crossroads",7:"More than four arms",
        8:"Private drive or entrance",9:"Other junction",13:"T or staggered junction",
        16:"Crossroads",17:"More than four arms",18:"Private drive or entrance",
        19:"Other junction",99:"Unknown junction detail",-1:"Missing / out of range"},
    "junction_control": {0:"Not at junction or no control",1:"Authorised person",2:"Automatic traffic signal",3:"Stop sign",4:"Give way / uncontrolled",-1:"Missing / out of range"},
    "light_conditions": {1:"Daylight",4:"Darkness - lights lit",5:"Darkness - lights unlit",6:"Darkness - no lighting",7:"Darkness - lighting unknown",-1:"Missing / out of range"},
    "weather_conditions": {1:"Fine - no high winds",2:"Raining - no high winds",3:"Snowing - no high winds",4:"Fine - high winds",5:"Raining - high winds",6:"Snowing - high winds",7:"Fog or mist",8:"Other weather",9:"Unknown weather",-1:"Missing / out of range"},
    "road_surface_conditions": {1:"Dry",2:"Wet or damp",3:"Snow",4:"Frost or ice",5:"Flood over 3 cm deep",9:"Unknown surface condition",-1:"Missing / out of range"},
    "special_conditions_at_site": {0:"None",1:"Auto traffic signal out",2:"Auto signal partly defective",3:"Road sign or marking defective / obscured",4:"Roadworks",5:"Road surface defective",6:"Oil or diesel",7:"Mud",-1:"Missing / out of range"},
    "carriageway_hazards": {0:"None",1:"Vehicle load on road",2:"Other object on road",3:"Previous accident",4:"Dog on road",5:"Other animal on road",6:"Pedestrian in carriageway",7:"Any animal in carriageway",-1:"Missing / out of range"},
    "urban_or_rural_area": {1:"Urban",2:"Rural",3:"Unallocated",-1:"Missing / out of range"},
    "first_road_class": {1:"Motorway",2:"A(M)",3:"A road",4:"B road",5:"C road",6:"Unclassified",-1:"Missing / out of range"},
    "second_road_class": {0:"Not at junction / no second road",1:"Motorway",2:"A(M)",3:"A road",4:"B road",5:"C road",6:"Unclassified",-1:"Missing / out of range"},
    "day_of_week": {1:"Sunday",2:"Monday",3:"Tuesday",4:"Wednesday",5:"Thursday",6:"Friday",7:"Saturday"},
    "month": {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"},
    "driver_sex_first": {1:"Male",2:"Female",3:"Not known / not applicable",9:"Unknown",-1:"Missing / out of range"},
    "vehicle_manoeuvre_first": {1:"Reversing",2:"Parked",3:"Waiting to go - held up",4:"Slowing or stopping",5:"Moving off",6:"U-turn",7:"Turning left",8:"Waiting to turn left",9:"Turning right",10:"Waiting to turn right",11:"Changing lane to left",12:"Changing lane to right",13:"Overtaking moving vehicle on offside",14:"Overtaking static vehicle on offside",15:"Overtaking on nearside",16:"Going ahead left-hand bend",17:"Going ahead right-hand bend",18:"Going ahead other",99:"Unknown manoeuvre",-1:"Missing / out of range"},
}
ADVANCED_BINARY = {
    "weekend_flag", "motorcycle_involved", "goods_vehicle_involved"
}


@st.cache_resource(show_spinner="Loading severity model...")
def load_prediction_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Missing model contract: {CONTRACT_PATH}")
    model = joblib.load(MODEL_PATH)
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not contract.get("features"):
        raise ValueError("model_contract.json contains no feature definitions")
    return model, contract


def prediction_default(spec):
    value = spec.get("default")
    if spec.get("dtype") == "numeric":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    return "Unknown" if value is None else str(value)


def prediction_input(spec):
    name = spec["feature"]
    label = PREDICTOR_LABELS.get(name, name.replace("_", " ").title())
    default = prediction_default(spec)

    # Decode the influential coded predictors for display, while returning
    # the exact numeric/categorical value expected by the saved model.
    if name == "vehicle_type_first":
        options = list(VEHICLE_TYPE_INPUT)
        default_code = int(float(default)) if str(default).replace(".", "", 1).lstrip("-").isdigit() else None
        default_label = VEHICLE_TYPE.get(default_code, "Unknown")
        index = options.index(default_label) if default_label in options else 0
        chosen = st.selectbox(label, options, index=index, key=f"pred_{name}")
        return VEHICLE_TYPE_INPUT[chosen]

    if name == "road_type":
        options = list(ROAD_TYPE_INPUT)
        default_code = int(float(default)) if str(default).replace(".", "", 1).lstrip("-").isdigit() else None
        default_label = ROAD_TYPE.get(default_code, "Unknown")
        index = options.index(default_label) if default_label in options else 0
        chosen = st.selectbox(label, options, index=index, key=f"pred_{name}")
        return ROAD_TYPE_INPUT[chosen]

    if name in {"vulnerable_road_user_involved", "trunk_road_flag"}:
        options = list(BINARY_INPUT)
        try:
            default_code = int(float(default))
        except (TypeError, ValueError):
            default_code = 0
        index = 1 if default_code == 1 else 0
        chosen = st.selectbox(label, options, index=index, key=f"pred_{name}")
        return BINARY_INPUT[chosen]

    if name in ADVANCED_CODE_LABELS:
        mapping = ADVANCED_CODE_LABELS[name]
        # Only offer codes that occur in the model contract examples when available;
        # otherwise use the validated mapping in model-code order.
        observed = []
        for raw in spec.get("category_examples", []):
            try:
                code = int(float(raw))
            except (TypeError, ValueError):
                continue
            if code in mapping and code not in observed:
                observed.append(code)
        codes = observed or list(mapping.keys())
        labels = [mapping[code] for code in codes]
        try:
            default_code = int(float(default))
        except (TypeError, ValueError):
            default_code = codes[0]
        index = codes.index(default_code) if default_code in codes else 0
        chosen = st.selectbox(label, labels, index=index, key=f"pred_{name}")
        return codes[labels.index(chosen)]

    if name in ADVANCED_BINARY:
        try:
            default_code = int(float(default))
        except (TypeError, ValueError):
            default_code = 0
        chosen = st.selectbox(label, list(BINARY_INPUT), index=1 if default_code == 1 else 0,
                              key=f"pred_{name}")
        return BINARY_INPUT[chosen]

    if name == "time_band":
        options = ["Night", "Morning peak", "Daytime", "Evening peak", "Late evening"]
        default_text = str(default)
        index = options.index(default_text) if default_text in options else 0
        return st.selectbox(label, options, index=index, key=f"pred_{name}")

    if spec.get("dtype") == "numeric":
        integers = {
            "speed_limit", "number_of_vehicles", "hour", "month", "day_of_week",
            "driver_age_min", "vulnerable_road_user_count", "vehicle_record_count",
            "motorcycle_involved", "goods_vehicle_involved",
            "vulnerable_road_user_involved", "weekend_flag",
        }
        if name in integers:
            return st.number_input(label, value=int(round(default)), step=1, key=f"pred_{name}")
        step = 100.0 if "engine_cc" in name else 1.0
        return st.number_input(label, value=float(default), step=step, key=f"pred_{name}")

    options = [str(x) for x in spec.get("category_examples", []) if x is not None]
    if str(default) not in options:
        options.insert(0, str(default))
    options = list(dict.fromkeys(options)) or ["Unknown"]
    return st.selectbox(label, options, index=options.index(str(default)), key=f"pred_{name}")

def render_prediction_module():
    st.markdown("### Collision Severity Prediction")
    st.caption(
        "Predicts severity after a collision has been recorded. "
        "It does not predict whether a collision will occur."
    )
    try:
        model, contract = load_prediction_assets()
    except Exception as exc:
        st.error(f"Prediction module unavailable: {exc}")
        st.info("Keep stats19_severity_model.joblib and model_contract.json in the models folder.")
        return

    specs = contract["features"]
    by_name = {spec["feature"]: spec for spec in specs}
    key_specs = [by_name[name] for name in KEY_PREDICTORS if name in by_name]
    key_names = {spec["feature"] for spec in key_specs}
    remaining = [spec for spec in specs if spec["feature"] not in key_names]

    with st.form("severity_prediction_form"):
        st.markdown("#### Key predictors")
        values = {}
        cols = st.columns(2)
        for i, spec in enumerate(key_specs):
            with cols[i % 2]:
                values[spec["feature"]] = prediction_input(spec)
        with st.expander("Advanced inputs: remaining model predictors"):
            cols = st.columns(2)
            for i, spec in enumerate(remaining):
                with cols[i % 2]:
                    values[spec["feature"]] = prediction_input(spec)
        submitted = st.form_submit_button("Predict severity", type="primary")

    if not submitted:
        st.info("Enter the collision characteristics and select Predict severity.")
        return

    row = {}
    for spec in specs:
        value = values.get(spec["feature"], prediction_default(spec))
        row[spec["feature"]] = float(value) if spec.get("dtype") == "numeric" else str(value)
    frame = pd.DataFrame([row], columns=[spec["feature"] for spec in specs])

    try:
        predicted = str(np.asarray(model.predict(frame)).reshape(-1)[0])
        probabilities = np.asarray(model.predict_proba(frame))[0]
        classes = [str(value) for value in model.classes_]
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    color = SEVERITY_COLORS.get(predicted, BLUE)
    st.markdown(
        f'<div class="kpi" style="--c:{color};max-width:520px">'
        f'<div class="label">Predicted collision severity</div>'
        f'<div class="value">{predicted}</div>'
        f'<div class="sub">Research-only CatBoost model</div></div>',
        unsafe_allow_html=True,
    )
    if len(classes) == len(probabilities):
        scores = pd.DataFrame({"Severity": classes, "Model score": probabilities})
        chart = alt.Chart(scores).mark_bar(cornerRadius=4).encode(
            x=alt.X("Model score:Q", axis=alt.Axis(format="%"), title=None),
            y=alt.Y("Severity:N", sort=SEVERITY_ORDER, title=None),
            color=alt.Color(
                "Severity:N",
                scale=alt.Scale(domain=SEVERITY_ORDER,
                                range=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER]),
                legend=None,
            ),
            tooltip=["Severity:N", alt.Tooltip("Model score:Q", format=".1%")],
        ).properties(height=150)
        show(chart)
        st.caption("Model scores are not calibrated personal-risk probabilities.")

    with st.expander("Model evidence and limitations"):
        st.warning(contract.get("operational_warning", "Further model improvement recommended"))
        st.write(contract.get("scope_warning", "Conditional severity prediction only."))
        st.write(
            "Untouched test: Macro-F1 0.3925; balanced accuracy 0.5677; "
            "Serious recall 0.4319; Fatal recall 0.6777; Fatal precision 0.0576."
        )


# Tabs
tab_dash, tab_pred, tab_rec = st.tabs(["📊 Dashboard", "🔮 Prediction", "📁 Collision records"])

with tab_dash:
    left, right = st.columns([3, 2])
    with left:
        with card("Where collisions happen"):
            draw_map(cur)
    with right:
        with card("Severity"):
            drill_donut(counts(view("severity")["collision_severity_label"].astype(str)), "severity", SEVERITY_COLORS)
        with card("Urban or rural"):
            drill_donut(counts(view("area")["area_label"].astype(str)), "area", AREA_COLORS)

    left, right = st.columns([3, 2])
    with left:
        with card("Collisions by month"):
            mv = view("month")["month_label"].astype(str).value_counts()
            months = sorted(x for x in mv.index if x != "Unknown")
            drill_bar(mv.rename_axis("label").reset_index(name="value"), "month", order=months)
    with right:
        with card("Fatal collisions per 1,000, by speed limit"):
            sr = speed_rates(view("speed"))
            if sr.empty:
                st.info("No speed-limit data.")
            else:
                drill_bar(sr[["label", "value", "n", "fatal"]], "speed", scale_color=True, order=list(sr["label"]),
                          vtitle="Fatal per 1,000", vfmt=".1f", extra={"n": "Collisions", "fatal": "Fatal"})

    left, right = st.columns([3, 2])
    with left:
        with card("When: day and hour"):
            hm = (cur[cur["hour_label"] != "Unknown"].groupby(["weekday", "hour_label"], observed=True)
                  .size().reset_index(name="n"))
            hm["weekday"], hm["hour_label"] = hm["weekday"].astype(str), hm["hour_label"].astype(str)
            show(alt.Chart(hm).mark_rect(cornerRadius=2).encode(
                x=alt.X("hour_label:O", sort=HOURS, title=None, axis=alt.Axis(labelAngle=-90)),
                y=alt.Y("weekday:O", sort=WEEKDAYS, title=None),
                color=alt.Color("n:Q", scale=alt.Scale(range=["#EAF0F8", BLUE, INK]), legend=None),
                tooltip=["weekday:O", "hour_label:O", alt.Tooltip("n:Q", format=",", title="Collisions")],
            ).properties(height=230))
    with right:
        with card("Who is most at risk? (% killed or seriously hurt)"):
            uc = (cas_for(view("user")).groupby("user_type").agg(n=("is_ksi", "size"), ksi=("is_ksi", "sum"))
                  .reset_index().rename(columns={"user_type": "label"}))
            uc = uc[uc["n"] > 0]
            uc["value"] = uc["ksi"] / uc["n"] * 100
            drill_bar(uc[["label", "value", "n"]], "user", scale_color=True, horizontal=True,
                      vtitle="% killed or serious", vfmt=".1f", extra={"n": "Casualties"})

    c1, c2, c3 = st.columns(3)
    with c1:
        with card("Severity by light"):
            lv = view("light").groupby(["light_label", "collision_severity_label"], observed=True).size().reset_index(name="n")
            lv.columns = ["label", "severity", "n"]
            lv["severity"] = lv["severity"].astype(str)
            drill_stack(lv, "light", LIGHT_ORDER, legend=True,
                        label_expr="replace(datum.label, 'Darkness - ', 'Dark: ')", height=235)
    with c2:
        with card("Severity by road type"):
            rv = view("road").groupby(["road_type_label", "collision_severity_label"], observed=True).size().reset_index(name="n")
            rv.columns = ["label", "severity", "n"]
            rv["severity"] = rv["severity"].astype(str)
            drill_stack(rv, "road", sorted(rv["label"].astype(str).unique()), legend=True, height=235)
    with c3:
        with card("Casualties by age"):
            ag = cas_cur.groupby(["age_band_label", "casualty_severity_label"], observed=True).size().reset_index(name="n")
            ag["age_band_label"] = ag["age_band_label"].astype(str)
            ag["casualty_severity_label"] = ag["casualty_severity_label"].astype(str)
            show(alt.Chart(ag).mark_bar().encode(
                x=alt.X("n:Q", title=None),
                y=alt.Y("age_band_label:N", sort=AGE_ORDER, title=None),
                color=alt.Color("casualty_severity_label:N", sort=SEVERITY_ORDER,
                                scale=alt.Scale(domain=SEVERITY_ORDER, range=[SEVERITY_COLORS[s] for s in SEVERITY_ORDER]),
                                legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["age_band_label:N", "casualty_severity_label:N", alt.Tooltip("n:Q", format=",", title="Casualties")],
            ).properties(height=235))

    c1, c2 = st.columns(2)
    with c1:
        with card("Collisions by day"):
            drill_bar(counts(view("weekday")["weekday"].astype(str)), "weekday", color=TEAL, order=WEEKDAYS, height=200)
    with c2:
        with card("Collisions by hour"):
            hv = view("hour")
            hv = hv[hv["hour_label"] != "Unknown"]
            drill_bar(counts(hv["hour_label"].astype(str)), "hour", color=AMBER, order=HOURS, height=200)

# Prediction module
with tab_pred:
    render_prediction_module()

# Records: pick one collision to see its vehicles and casualties
with tab_rec:
    cols = ["collision_index", "date", "time", "collision_severity_label", "number_of_vehicles",
            "number_of_casualties", "speed_label", "road_type_label", "light_label", "area_label"]
    table = cur.sort_values("date", ascending=False).head(5000)
    shown = table[[c for c in cols if c in table.columns]].rename(columns={
        "collision_index": "Collision ID", "date": "Date", "time": "Time",
        "collision_severity_label": "Severity", "number_of_vehicles": "Vehicles",
        "number_of_casualties": "Casualties", "speed_label": "Speed limit",
        "road_type_label": "Road type", "light_label": "Light", "area_label": "Area"})
    st.caption(f"Latest {len(table):,} of {len(cur):,} collisions" + (" · select a row for details" if ROWSEL else ""))
    cid = None
    if ROWSEL:
        event = st.dataframe(shown, hide_index=True, on_select="rerun", selection_mode="single-row",
                             key="records_table", **_wide(_DF))
        rows = event.selection.rows if event and event.selection else []
        if rows and rows[0] < len(table):
            cid = table.iloc[rows[0]]["collision_index"]
    else:
        st.dataframe(shown, hide_index=True, **_wide(_DF))
        cid = st.selectbox("Collision ID", ["—"] + list(table["collision_index"].head(300)))
        cid = None if cid == "—" else cid
    if cid:
        r = table[table["collision_index"] == cid].iloc[0]
        st.markdown(f"#### Collision {cid}")
        st.write(f"**{r['collision_severity_label']}** · {r['date']:%d %b %Y} {r['time']} · "
                 f"{r['area_label']} · {r['road_type_label']} · {r['speed_label']} mph · {r['light_label']}")
        v1, v2 = st.columns(2)
        with v1:
            with card("Vehicles"):
                vv = vehicles[vehicles["collision_index"] == cid]
                vcols = [c for c in ["vehicle_type_label", "driver_age_band_label", "driver_sex_label"] if c in vv.columns]
                st.dataframe(vv[vcols].rename(columns={"vehicle_type_label": "Vehicle", "driver_age_band_label": "Driver age",
                                                       "driver_sex_label": "Driver sex"}), hide_index=True)
        with v2:
            with card("Casualties"):
                cc = casualties[casualties["collision_index"] == cid]
                st.dataframe(cc[["user_type", "age_band_label", "sex_label", "casualty_severity_label"]].rename(
                    columns={"user_type": "Road user", "age_band_label": "Age", "sex_label": "Sex",
                             "casualty_severity_label": "Severity"}), hide_index=True)
    st.download_button("⬇️ Download these records (CSV)", data=shown.to_csv(index=False).encode("utf-8"),
                       file_name="road_collisions_selection.csv", mime="text/csv")

st.caption("Source: UK Department for Transport road casualty statistics.")