import inspect
from contextlib import contextmanager
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UK Road Safety Dashboard", page_icon="🚦", layout="wide")

# ---------------------------------------------------------
# Compatibility: works on older Streamlit, and switches on
# click-to-filter charts when the installed version supports it
# ---------------------------------------------------------
_ALT = inspect.signature(st.altair_chart).parameters
_DF = inspect.signature(st.dataframe).parameters
CLICK = "on_select" in _ALT            # click a chart to filter
ROWSEL = "on_select" in _DF            # click a table row to drill in
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


# ---------------------------------------------------------
# Data location: finds the CSVs by keyword, so filenames can differ
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Palette
# ---------------------------------------------------------
INK, BLUE, TEAL = "#12263F", "#1D4E89", "#2A9D8F"
AMBER, RED, GREEN, GREY = "#F2A900", "#C8102E", "#2E9E6B", "#8A97A8"

SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]
SEVERITY_COLORS = {"Fatal": RED, "Serious": AMBER, "Slight": GREEN}
SEVERITY_RGB = {"Fatal": [200, 16, 46, 230], "Serious": [242, 169, 0, 200], "Slight": [46, 158, 107, 150]}
AREA_COLORS = {"Urban": BLUE, "Rural": TEAL, "Unallocated": GREY, "Unknown": GREY}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
HOURS = [f"{h:02d}:00" for h in range(24)]

# ---------------------------------------------------------
# Lookups
# ---------------------------------------------------------
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
    fatal = int(coll["is_fatal"].sum())
    ksi = int(cas["is_ksi"].sum())
    return dict(
        n=n, nc=nc, fatal=fatal, ksi=ksi,
        fatal_rate=fatal / n * 1000 if n else 0,
        ksi_pct=ksi / nc * 100 if nc else 0,
        vru_pct=cas["is_vru"].mean() * 100 if nc else 0,
        dark_pct=coll["is_dark"].mean() * 100 if n else 0,
    )


@st.cache_resource
def baseline():
    c, ca, _ = load_data()
    return metrics(c, ca)


# ---------------------------------------------------------
# Styling: light page, white cards
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');
    html, body, [class*="st-"] {{ font-family: 'Barlow', sans-serif; }}
    .stApp {{ background: #EDF1F6; }}
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
    .kpi {{ background: #fff; border-radius: 12px; padding: .7rem 1rem; height: 100%;
            border-left: 6px solid var(--c); box-shadow: 0 1px 4px rgba(18,38,63,.12); }}
    .kpi .label {{ color: #5B6B80; font-size: .85rem; font-weight: 600; }}
    .kpi .value {{ color: var(--c); font-family: 'Barlow Condensed', sans-serif;
                   font-size: 2.2rem; font-weight: 700; line-height: 1.1; }}
    .kpi .sub {{ color: #5B6B80; font-size: .8rem; }}
    .up {{ color: {RED}; font-weight: 600; }} .down {{ color: {GREEN}; font-weight: 600; }}
    .flat {{ color: #7A8799; }}
    .chip {{ display: inline-block; background: {BLUE}; color: #fff; border-radius: 999px;
             padding: .1rem .7rem; margin: 0 .3rem .3rem 0; font-size: .85rem; }}
    .legend span {{ margin-right: .9rem; font-size: .85rem; color: #44546A; }}
    section[data-testid="stSidebar"] {{ border-right: 4px solid {AMBER}; }}
    button[data-baseweb="tab"] p {{ font-weight: 600; font-size: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
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

st.markdown(f'<div class="topbar"><span class="t">🚦 UK Road Safety Dashboard</span>'
            f'<span class="p">{period}</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
severity_options = SEVERITY_ORDER
area_options = sorted(collisions["area_label"].dropna().unique().tolist())
speed_options = sorted((s for s in collisions["speed_label"].unique() if s != "Unknown"), key=int)
if (collisions["speed_label"] == "Unknown").any():
    speed_options.append("Unknown")
road_options = sorted(collisions["road_type_label"].dropna().unique().tolist())

DEFAULTS = {"f_dates": (min_date, max_date), "f_severity": severity_options,
            "f_areas": area_options, "f_speeds": speed_options, "f_roads": road_options}
for k, v in DEFAULTS.items():
    if k != "f_dates":
        st.session_state.setdefault(k, v)
st.session_state.setdefault("ver", 0)


def reset_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["ver"] += 1


def clear_drill():
    st.session_state["ver"] += 1


st.sidebar.title("🔎 Filters")
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date, key="f_dates")
st.sidebar.multiselect("Severity", severity_options, key="f_severity")
st.sidebar.multiselect("Area", area_options, key="f_areas")
st.sidebar.multiselect("Speed limit (mph)", speed_options, key="f_speeds")
st.sidebar.multiselect("Road type", road_options, key="f_roads")

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]
else:
    start_date = end_date = date_range

base = collisions[
    (collisions["date"] >= pd.Timestamp(start_date))
    & (collisions["date"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
    & collisions["collision_severity_label"].isin(st.session_state["f_severity"])
    & collisions["area_label"].isin(st.session_state["f_areas"])
    & collisions["speed_label"].isin(st.session_state["f_speeds"])
    & collisions["road_type_label"].isin(st.session_state["f_roads"])
]
st.sidebar.divider()
st.sidebar.write(f"**{len(base):,}** collisions match")
st.sidebar.button("Reset filters", on_click=reset_filters, type="primary")
if not CLICK:
    st.sidebar.info("For click-to-filter charts, run:\n\n`python -m pip install -U streamlit altair`")

if base.empty:
    st.warning("No records match these filters.")
    st.stop()

# ---------------------------------------------------------
# Drill-through state (clicks on charts)
# ---------------------------------------------------------
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
    """Data for a chart: every click applied except the chart's own."""
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

# ---------------------------------------------------------
# KPI cards (compared with the whole dataset)
# ---------------------------------------------------------
m, b = metrics(cur, cas_cur), baseline()
days = max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1, 1)


def delta(val, ref, unit=" pts"):
    d = val - ref
    if abs(d) < 0.05:
        return '<span class="flat">= all data</span>'
    cls, arrow = ("up", "▲") if d > 0 else ("down", "▼")
    return f'<span class="{cls}">{arrow} {abs(d):.1f}{unit} vs all data</span>'


def kpi(col, label, value, sub, color):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


k = st.columns(5)
kpi(k[0], "Collisions", f"{m['n']:,}", f"≈ {m['n'] / days:,.0f} a day · {m['nc']:,} casualties", BLUE)
kpi(k[1], "Killed or seriously injured", f"{m['ksi']:,}",
    f"{m['ksi_pct']:.1f}% of casualties · {delta(m['ksi_pct'], b['ksi_pct'])}", AMBER)
kpi(k[2], "Fatal collisions", f"{m['fatal']:,}",
    f"{m['fatal_rate']:.1f} per 1,000 · {delta(m['fatal_rate'], b['fatal_rate'], '')}", RED)
kpi(k[3], "Pedestrians, cyclists, motorcyclists", f"{m['vru_pct']:.0f}%",
    f"of casualties · {delta(m['vru_pct'], b['vru_pct'])}", TEAL)
kpi(k[4], "Collisions in the dark", f"{m['dark_pct']:.0f}%",
    f"of collisions · {delta(m['dark_pct'], b['dark_pct'])}", INK)
st.write("")


# ---------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------
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
# Tabs
# ---------------------------------------------------------
tab_dash, tab_rec = st.tabs(["📊 Dashboard", "📁 Collision records"])

with tab_dash:
    # Row 1: map + severity / area donuts
    left, right = st.columns([3, 2])
    with left:
        with card("Where collisions happen"):
            draw_map(cur)
    with right:
        with card("Severity"):
            drill_donut(counts(view("severity")["collision_severity_label"].astype(str)), "severity", SEVERITY_COLORS)
        with card("Urban or rural"):
            drill_donut(counts(view("area")["area_label"].astype(str)), "area", AREA_COLORS)

    # Row 2: trend + speed
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

    # Row 3: when + who
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

    # Row 4: conditions
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

    # Row 5: day of week + hour (clickable)
    c1, c2 = st.columns(2)
    with c1:
        with card("Collisions by day"):
            drill_bar(counts(view("weekday")["weekday"].astype(str)), "weekday", color=TEAL, order=WEEKDAYS, height=200)
    with c2:
        with card("Collisions by hour"):
            hv = view("hour")
            hv = hv[hv["hour_label"] != "Unknown"]
            drill_bar(counts(hv["hour_label"].astype(str)), "hour", color=AMBER, order=HOURS, height=200)

# ---------------------------------------------------------
# Records: pick one collision to see its vehicles and casualties
# ---------------------------------------------------------
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