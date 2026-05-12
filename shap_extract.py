"""
shap_extract.py

Streamlit web tool for extracting OSM roads within a radius,
filtering by type, calculating distances, comparing road networks
across different dates, and exporting as Shapefile.

Run with:
    streamlit run shap_extract.py

Install requirements using venv (macOS/Linux):

    python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Install requirements using venv (Windows):

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

For new terminal run this first:
    source .venv/bin/activate (macOS/Linux)
    .venv\Scripts\activate (Windows)
"""

# Imports required 
import io
import zipfile
import tempfile
import os
import datetime
import requests

import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points

# Page config
st.set_page_config(page_title="OSM Road Extractor", page_icon="🛣️", layout="wide")

st.title("🛣️ OSM Road Extractor")
st.caption("Fetch roads from OpenStreetMap at any date, filter by type, measure distances, and export as Shapefile.")

# Colour config

HIGHWAY_COLORS = {
    "motorway":    "#e06030",
    "trunk":       "#e06030",
    "primary":     "#d4873a",
    "secondary":   "#7F77DD",
    "tertiary":    "#378ADD",
    "residential": "#639922",
    "service":     "#888780",
    "footway":     "#1D9E75",
    "cycleway":    "#0F6E56",
    "path":        "#9FE1CB",
}

def get_color(highway_val):
    if isinstance(highway_val, list):
        highway_val = highway_val[0]
    return HIGHWAY_COLORS.get(str(highway_val), "#888780")


def shp_safe(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Truncate column names to 10 chars and flatten list values for Shapefile compat."""
    out = gdf.copy()
    seen, rename_map = set(), {}
    for col in out.columns:
        if col == "geometry":
            continue
        short, counter = col[:10], 1
        cand = short
        while cand in seen:
            suf = str(counter)
            cand = short[:10 - len(suf)] + suf
            counter += 1
        if cand != col:
            rename_map[col] = cand
        seen.add(cand)
    out = out.rename(columns=rename_map)
    for col in out.columns:
        if col == "geometry":
            continue
        if out[col].apply(lambda x: isinstance(x, list)).any():
            out[col] = out[col].apply(lambda x: "|".join(map(str, x)) if isinstance(x, list) else x)
    return out


def overpass_roads(lat: float, lon: float, radius_m: float,
                   date_str: str = None) -> gpd.GeoDataFrame:
    """
    Query Overpass API for highway ways around a point.
    date_str: ISO8601 like "2020-06-01T00:00:00Z", or None for current.
    Returns GeoDataFrame of LineStrings in EPSG:4326.
    Retries across two mirror servers with exponential back-off.
    """
    import time

    MIRRORS = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    date_clause = f'[date:"{date_str}"]' if date_str else ""
    query = f"""
    [out:json]{date_clause}[timeout:90][maxsize:536870912];
    (
      way["highway"](around:{radius_m},{lat},{lon});
    );
    (._;>;);
    out geom;
    """

    last_err = None
    for mirror in MIRRORS:
        for attempt in range(3):          # up to 3 retries per mirror
            try:
                wait = 2 ** attempt       # 1s, 2s, 4s back-off
                if attempt > 0:
                    time.sleep(wait)
                resp = requests.post(mirror, data={"data": query}, timeout=120)
                if resp.status_code == 429:
                    time.sleep(10)        # rate-limited — wait longer
                    continue
                if resp.status_code in (406, 504, 502, 503):
                    time.sleep(5)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break                     # success — exit retry loop
            except requests.exceptions.Timeout:
                last_err = "Request timed out"
                continue
            except Exception as e:
                last_err = str(e)
                continue
        else:
            continue                      # this mirror failed all retries, try next
        break                             # got a good response
    else:
        raise RuntimeError(
            f"All Overpass mirrors failed. Last error: {last_err}\n"
            "Tips: reduce radius, use a coarser interval, or try again in a few minutes."
        )

    nodes = {
        el["id"]: (el["lon"], el["lat"])
        for el in data["elements"] if el["type"] == "node"
    }

    rows = []
    for el in data["elements"]:
        if el["type"] != "way":
            continue
        if "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
        else:
            coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
        if len(coords) < 2:
            continue
        tags = el.get("tags", {})
        rows.append({
            "osm_id":   el["id"],
            "name":     tags.get("name"),
            "highway":  tags.get("highway"),
            "oneway":   tags.get("oneway"),
            "lanes":    tags.get("lanes"),
            "maxspeed": tags.get("maxspeed"),
            "surface":  tags.get("surface"),
            "geometry": LineString(coords),
        })

    if not rows:
        return gpd.GeoDataFrame(
            columns=["osm_id","name","highway","oneway","lanes","maxspeed","surface","geometry"],
            crs="EPSG:4326"
        )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def add_roads_to_map(fmap, gdf, color_by_type=True,
                     default_color="#7F77DD", weight=2, opacity=0.8):
    for _, row in gdf.iterrows():
        hw  = row.get("highway") or "unknown"
        nm  = row.get("name") or ""
        col = get_color(hw) if color_by_type else default_color
        tip = f"{hw}" + (f" — {nm}" if nm and str(nm) != "nan" else "")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda f, c=col: {"color": c, "weight": weight, "opacity": opacity},
            tooltip=tip,
        ).add_to(fmap)


def make_zip_shp(gdf, stem):
    """Package a GeoDataFrame as a zipped Shapefile, return BytesIO or None."""
    if gdf is None or gdf.empty:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        shp_safe(gdf).to_file(os.path.join(tmp, f"{stem}.shp"), driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for ext in [".shp", ".dbf", ".shx", ".prj", ".cpg"]:
                fp = os.path.join(tmp, f"{stem}{ext}")
                if os.path.exists(fp):
                    zf.write(fp, f"{stem}{ext}")
        buf.seek(0)
        return buf


# Preset loader

PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets.csv")

def load_presets(path: str) -> list[dict]:
    """
    Load preset coordinates from a CSV file with columns: lat, lon, date (optional).
    Skips comment lines (starting with #) and blank rows.
    Returns list of dicts with keys: label, lat, lon, date.
    """
    import csv
    presets = []
    if not os.path.exists(path):
        return presets
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(
            (row for row in f if not row.strip().startswith("#") and row.strip()),
        )
        for i, row in enumerate(reader):
            try:
                lat  = float(row["lat"].strip())
                lon  = float(row["lon"].strip())
                date = row.get("date", "").strip() or None
                presets.append({
                    "label": f"{i+1}. {lat:.5f}, {lon:.5f}" + (f"  [{date}]" if date else ""),
                    "lat":   lat,
                    "lon":   lon,
                    "date":  date,
                })
            except (ValueError, KeyError):
                continue
    return presets[:50]


def preset_date_to_query(date_str: str | None):
    """Convert YYYY-MM-DD string to Overpass date clause, or None for current."""
    if not date_str:
        return None, "current"
    try:
        d = datetime.date.fromisoformat(date_str)
        return f"{d.isoformat()}T00:00:00Z", str(d)
    except ValueError:
        return None, "current"


# Sidebar 
with st.sidebar:
    st.header("⚙️ Parameters")

    # Presets
    presets = load_presets(PRESET_FILE)

    if presets:
        st.subheader("📌 Presets")
        preset_labels = ["— select a preset —"] + [p["label"] for p in presets]
        chosen_label  = st.selectbox("Load preset", preset_labels, index=0)

        if chosen_label != "— select a preset —":
            preset = next(p for p in presets if p["label"] == chosen_label)
            st.session_state["_preset_lat"]  = preset["lat"]
            st.session_state["_preset_lon"]  = preset["lon"]
            st.session_state["_preset_date"] = preset["date"]
        else:
            for k in ["_preset_lat","_preset_lon","_preset_date"]:
                st.session_state.pop(k, None)
        st.divider()
    else:
        st.caption(
            "No presets loaded. Place a `presets.csv` file next to this script. "
            "Download the template from the **📌 Presets** tab."
        )
        st.divider()

    # Manual coordinate inputs (pre-filled by preset if one was chosen)
    st.subheader("📍 Coordinates")
    lat    = st.number_input("Latitude",  value=st.session_state.get("_preset_lat",  15.5527), format="%.6f")
    lon    = st.number_input("Longitude", value=st.session_state.get("_preset_lon",  32.5324), format="%.6f")
    radius = st.slider("Search radius (m)", 100, 5000, 1000, step=100)

    st.divider()
    st.subheader("📅 Date")
    use_date = st.toggle("Query a specific date", value=False)

    MIN_DATE = datetime.date(2012, 9, 12)
    MAX_DATE = datetime.date.today() - datetime.timedelta(days=1)

    query_date  = None
    date_label  = "current"

    # Default date from preset (if available)
    _preset_date_val = st.session_state.get("_preset_date")
    _default_date = MAX_DATE
    if _preset_date_val:
        try:
            _default_date = datetime.date.fromisoformat(_preset_date_val)
            if not use_date:
                st.caption(f"Preset has date **{_preset_date_val}** — enable toggle to use it")
        except ValueError:
            pass

    if use_date:
        selected_date = st.date_input(
            "Road network as of",
            value=_default_date,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            help="Overpass historical data available from September 2012 onwards.",
        )
        query_date = f"{selected_date.isoformat()}T00:00:00Z"
        date_label = str(selected_date)
        st.caption(f"Will query OSM state on **{selected_date}**")
    else:
        st.caption("Querying **current** OSM data")

    st.divider()
    fetch_btn = st.button("🔍 Fetch Roads", use_container_width=True, type="primary")

# Session state
if "gdf"    not in st.session_state: st.session_state.gdf    = None
if "params" not in st.session_state: st.session_state.params = {}

# Fetch roads
if fetch_btn:
    with st.spinner(f"Querying Overpass API — {radius} m, {date_label}…"):
        try:
            result = overpass_roads(lat, lon, radius, query_date)
            if result.empty:
                st.warning("No roads returned. Try a larger radius or different location.")
            else:
                st.session_state.gdf    = result
                st.session_state.params = {
                    "lat": lat, "lon": lon, "radius": radius,
                    "date": query_date, "date_label": date_label,
                }
                st.success(f"✅ {len(result):,} road segments ({date_label})")
        except Exception as e:
            st.error(f"Overpass query failed: {e}")

# Main
if st.session_state.gdf is not None:
    gdf = st.session_state.gdf.copy()
    p   = st.session_state.params

    tab_map, tab_nearest, tab_compare, tab_interval, tab_filter, tab_stats, tab_export, tab_presets = st.tabs(
        ["🗺️ Map", "📍 Nearest Road", "🕐 Compare Dates", "📈 Interval Analysis",
         "🔎 Filter & Select", "📊 Statistics", "💾 Export", "📌 Presets"]
    )

    # Map
    with tab_map:
        st.caption(f"Showing roads — **{p['date_label']}**")
        m = folium.Map(location=[p["lat"], p["lon"]], zoom_start=14, tiles="CartoDB positron")
        folium.Circle(location=[p["lat"], p["lon"]], radius=p["radius"],
                      color="#7F77DD", fill=True, fill_opacity=0.05,
                      weight=2, dash_array="6 4").add_to(m)
        folium.Marker(location=[p["lat"], p["lon"]], tooltip="Centre point",
                      icon=folium.Icon(color="purple", icon="crosshairs", prefix="fa")).add_to(m)
        add_roads_to_map(m, gdf)
        st_folium(m, width="100%", height=500, returned_objects=[])

    # Nearest Road
    with tab_nearest:
        st.subheader("Find nearest road from a coordinate")
        st.caption("Finds the closest road segment and draws a connector with the distance.")

        col_a, col_b = st.columns(2)
        with col_a:
            q_lat = st.number_input("Query latitude",  value=p["lat"], format="%.6f", key="q_lat")
        with col_b:
            q_lon = st.number_input("Query longitude", value=p["lon"], format="%.6f", key="q_lon")

        with st.expander("Convert DMS → decimal"):
            dc1, dc2 = st.columns(2)
            with dc1:
                st.markdown("**Latitude**")
                d1   = st.number_input("Degrees",  0,   90,  14,    key="d1")
                m1   = st.number_input("Minutes",  0,   59,  46,    key="m1")
                s1   = st.number_input("Seconds",  0.0, 59.99, 14.06, format="%.2f", key="s1")
                dir1 = st.radio("Direction", ["N","S"], horizontal=True, key="dir1")
            with dc2:
                st.markdown("**Longitude**")
                d2   = st.number_input("Degrees",  0,   180, 33,    key="d2")
                m2   = st.number_input("Minutes",  0,   59,  20,    key="m2")
                s2   = st.number_input("Seconds",  0.0, 59.99, 36.83, format="%.2f", key="s2")
                dir2 = st.radio("Direction", ["E","W"], horizontal=True, key="dir2")
            dd_lat = round(d1 + m1/60 + s1/3600, 6) * (-1 if dir1 == "S" else 1)
            dd_lon = round(d2 + m2/60 + s2/3600, 6) * (-1 if dir2 == "W" else 1)
            st.info(f"Decimal degrees → **{dd_lat}, {dd_lon}**")
            if st.button("Use these coordinates"):
                st.session_state["q_lat"] = dd_lat
                st.session_state["q_lon"] = dd_lon
                st.rerun()

        # Find nearest road
        if st.button("📍 Find nearest road", type="primary"):
            qpt      = Point(q_lon, q_lat)
            gdf_m    = gdf.to_crs(epsg=3857)
            qpt_m    = gpd.GeoSeries([qpt], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
            dists    = gdf_m.geometry.distance(qpt_m)
            nidx     = dists.idxmin()
            dist_m   = dists[nidx]
            nrow     = gdf.loc[nidx]
            _, cpt_m = nearest_points(qpt_m, gdf_m.loc[nidx, "geometry"])
            cpt_wgs  = gpd.GeoSeries([cpt_m], crs="EPSG:3857").to_crs(epsg=4326).iloc[0]

            hw = nrow.get("highway") or "unknown"
            if isinstance(hw, list): hw = hw[0]
            name = nrow.get("name")
            if isinstance(name, list): name = name[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Distance to road", f"{dist_m:.1f} m")
            c2.metric("Road type",        str(hw))
            c3.metric("Road name",        str(name) if name and str(name) != "nan" else "—")

            mid_lat = (q_lat + cpt_wgs.y) / 2
            mid_lon = (q_lon + cpt_wgs.x) / 2

            nm = folium.Map(location=[mid_lat, mid_lon], zoom_start=16, tiles="CartoDB positron")
            add_roads_to_map(nm, gdf, color_by_type=False,
                             default_color="#b0b0b0", weight=1.5, opacity=0.5)
            folium.GeoJson(nrow.geometry.__geo_interface__,
                           style_function=lambda f: {"color":"#e06030","weight":4,"opacity":1},
                           tooltip=f"{hw}" + (f" — {name}" if name and str(name) != "nan" else "")
                           ).add_to(nm)
            folium.Marker(location=[q_lat, q_lon],
                          tooltip=f"Query point ({q_lat:.5f}, {q_lon:.5f})",
                          icon=folium.Icon(color="purple", icon="map-marker", prefix="fa")).add_to(nm)
            folium.CircleMarker(location=[cpt_wgs.y, cpt_wgs.x], radius=6,
                                color="#e06030", fill=True, fill_color="#e06030",
                                tooltip="Closest point on road").add_to(nm)
            folium.PolyLine(locations=[[q_lat, q_lon],[cpt_wgs.y, cpt_wgs.x]],
                            color="#7F77DD", weight=2.5, dash_array="6 4",
                            tooltip=f"{dist_m:.1f} m").add_to(nm)
            folium.Marker(
                location=[mid_lat, mid_lon],
                icon=folium.DivIcon(
                    html=f'<div style="background:#7F77DD;color:#fff;padding:2px 7px;'
                         f'border-radius:10px;font-size:11px;font-weight:600;'
                         f'white-space:nowrap">{dist_m:.1f} m</div>',
                    icon_size=(80, 20), icon_anchor=(40, 10),
                ),
            ).add_to(nm)
            st_folium(nm, width="100%", height=460, returned_objects=[])

            with st.expander("Full attributes of nearest road segment"):
                attr = {k: v for k, v in nrow.items() if k != "geometry"}
                st.json({k: (v[0] if isinstance(v, list) else str(v)) for k, v in attr.items()})

    # Compare Dates
    with tab_compare:
        st.subheader("Compare road network across two dates")
        st.caption(
            "Fetches the same area at two different dates and highlights roads "
            "that were **added**, **removed**, or **unchanged** between them."
        )

        cc1, cc2 = st.columns(2)
        with cc1:
            date_a = st.date_input("Date A (earlier)", value=datetime.date(2020, 1, 1),
                                   min_value=MIN_DATE, max_value=MAX_DATE, key="cmp_a")
        with cc2:
            date_b = st.date_input("Date B (later)",   value=MAX_DATE,
                                   min_value=MIN_DATE, max_value=MAX_DATE, key="cmp_b")

        if st.button("🔄 Run comparison", type="primary"):
            if date_a >= date_b:
                st.error("Date A must be earlier than Date B.")
            else:
                with st.spinner(f"Fetching roads on {date_a}…"):
                    gdf_a = overpass_roads(p["lat"], p["lon"], p["radius"],
                                           f"{date_a.isoformat()}T00:00:00Z")
                with st.spinner(f"Fetching roads on {date_b}…"):
                    gdf_b = overpass_roads(p["lat"], p["lon"], p["radius"],
                                           f"{date_b.isoformat()}T00:00:00Z")

                ids_a = set(gdf_a["osm_id"].astype(str))
                ids_b = set(gdf_b["osm_id"].astype(str))

                gdf_added   = gdf_b[gdf_b["osm_id"].astype(str).isin(ids_b - ids_a)]
                gdf_removed = gdf_a[gdf_a["osm_id"].astype(str).isin(ids_a - ids_b)]
                gdf_kept    = gdf_b[gdf_b["osm_id"].astype(str).isin(ids_a & ids_b)]

                def total_km(g):
                    if g.empty: return 0.0
                    return round(g.to_crs(epsg=3857).geometry.length.sum() / 1000, 2)

                # Metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(f"Roads on {date_a}", f"{len(gdf_a):,}")
                m2.metric(f"Roads on {date_b}", f"{len(gdf_b):,}")
                m3.metric("➕ Added",   f"{len(gdf_added):,}",
                          delta=f"+{len(gdf_added)}", delta_color="normal")
                m4.metric("➖ Removed", f"{len(gdf_removed):,}",
                          delta=f"-{len(gdf_removed)}", delta_color="inverse")

                st.divider()

                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Length added",   f"{total_km(gdf_added)} km")
                lc2.metric("Length removed", f"{total_km(gdf_removed)} km")
                lc3.metric("Net change",
                           f"{round(total_km(gdf_added) - total_km(gdf_removed), 2)} km")

                st.divider()

                # Map
                ctr = [p["lat"], p["lon"]]
                cm  = folium.Map(location=ctr, zoom_start=14, tiles="CartoDB positron")
                folium.Circle(location=ctr, radius=p["radius"],
                              color="#888", fill=False, weight=1, dash_array="4 4").add_to(cm)

                for _, row in gdf_kept.iterrows():
                    folium.GeoJson(row.geometry.__geo_interface__,
                                   style_function=lambda f: {"color":"#b0b0b0","weight":1.5,"opacity":0.6},
                                   tooltip="Unchanged").add_to(cm)

                for _, row in gdf_added.iterrows():
                    hw_ = row.get("highway") or ""
                    nm_ = row.get("name")    or ""
                    folium.GeoJson(row.geometry.__geo_interface__,
                                   style_function=lambda f: {"color":"#1D9E75","weight":3,"opacity":0.9},
                                   tooltip=f"➕ Added | {hw_}" + (f" — {nm_}" if nm_ and str(nm_) != "nan" else "")
                                   ).add_to(cm)

                for _, row in gdf_removed.iterrows():
                    hw_ = row.get("highway") or ""
                    nm_ = row.get("name")    or ""
                    folium.GeoJson(row.geometry.__geo_interface__,
                                   style_function=lambda f: {"color":"#E24B4A","weight":3,"opacity":0.9},
                                   tooltip=f"➖ Removed | {hw_}" + (f" — {nm_}" if nm_ and str(nm_) != "nan" else "")
                                   ).add_to(cm)

                legend_html = """
                <div style="position:fixed;bottom:20px;left:20px;z-index:1000;
                            background:white;padding:10px 14px;border-radius:8px;
                            border:1px solid #ccc;font-size:12px;line-height:1.9">
                  <b>Legend</b><br>
                  <span style="color:#1D9E75;font-weight:700">━━</span> Added<br>
                  <span style="color:#E24B4A;font-weight:700">━━</span> Removed<br>
                  <span style="color:#b0b0b0;font-weight:700">━━</span> Unchanged
                </div>"""
                cm.get_root().html.add_child(folium.Element(legend_html))
                st_folium(cm, width="100%", height=500, returned_objects=[])

                st.divider()
                st.markdown("**Export comparison results**")
                ec1, ec2 = st.columns(2)
                with ec1:
                    buf = make_zip_shp(gdf_added, "roads_added")
                    if buf:
                        st.download_button("⬇️ Download added roads (.zip)",
                                           data=buf, file_name="roads_added.zip",
                                           mime="application/zip")
                    else:
                        st.info("No added roads to export.")
                with ec2:
                    buf = make_zip_shp(gdf_removed, "roads_removed")
                    if buf:
                        st.download_button("⬇️ Download removed roads (.zip)",
                                           data=buf, file_name="roads_removed.zip",
                                           mime="application/zip")
                    else:
                        st.info("No removed roads to export.")

    # Time-Interval Analysis
    with tab_interval:
        st.subheader("Road network change over a time interval")
        st.caption(
            "Samples the road network at regular intervals between two dates. "
            "Tracks total length, segment count, and net additions/removals at each snapshot."
        )

        ia1, ia2 = st.columns(2)
        with ia1:
            iv_start = st.date_input("Start date", value=datetime.date(2016, 1, 1),
                                     min_value=MIN_DATE, max_value=MAX_DATE, key="iv_start")
        with ia2:
            iv_end = st.date_input("End date", value=MAX_DATE,
                                   min_value=MIN_DATE, max_value=MAX_DATE, key="iv_end")

        interval_unit = st.select_slider(
            "Snapshot interval",
            options=["Monthly", "Quarterly", "Every 6 months", "Yearly"],
            value="Yearly",
        )

        INTERVAL_MONTHS = {
            "Monthly": 1, "Quarterly": 3, "Every 6 months": 6, "Yearly": 12
        }

        def generate_snapshot_dates(start, end, step_months):
            dates, cur = [], start
            while cur <= end:
                dates.append(cur)
                m = cur.month - 1 + step_months
                cur = cur.replace(year=cur.year + m // 12, month=m % 12 + 1)
            if dates[-1] != end:
                dates.append(end)
            return dates

        step = INTERVAL_MONTHS[interval_unit]
        preview_dates = generate_snapshot_dates(iv_start, iv_end, step)
        st.caption(f"Will fetch **{len(preview_dates)} snapshots**: "
                   f"{', '.join(str(d) for d in preview_dates[:5])}"
                   + (" …" if len(preview_dates) > 5 else ""))

        if len(preview_dates) > 20:
            st.warning("More than 20 snapshots — consider a coarser interval to avoid rate limiting.")

        if st.button("▶️ Run interval analysis", type="primary"):
            if iv_start >= iv_end:
                st.error("Start date must be before end date.")
            else:
                records   = []
                prev_ids  = None
                progress  = st.progress(0, text="Fetching snapshots…")

                for i, snap_date in enumerate(preview_dates):
                    if i > 0:
                        import time; time.sleep(2)   # Overpass delay for rate limits
                    progress.progress(
                        int((i + 1) / len(preview_dates) * 100),
                        text=f"Fetching {snap_date} ({i+1}/{len(preview_dates)})…"
                    )
                    try:
                        snap_gdf = overpass_roads(
                            p["lat"], p["lon"], p["radius"],
                            f"{snap_date.isoformat()}T00:00:00Z"
                        )
                        cur_ids   = set(snap_gdf["osm_id"].astype(str))
                        total_km  = round(
                            snap_gdf.to_crs(epsg=3857).geometry.length.sum() / 1000, 3
                        ) if not snap_gdf.empty else 0.0
                        n_segs    = len(snap_gdf)
                        added     = len(cur_ids - prev_ids) if prev_ids is not None else 0
                        removed   = len(prev_ids - cur_ids) if prev_ids is not None else 0
                        net       = added - removed

                        # Road type breakdown
                        if not snap_gdf.empty and "highway" in snap_gdf.columns:
                            hw_flat   = snap_gdf["highway"].apply(
                                lambda x: x[0] if isinstance(x, list) else x
                            )
                            hw_counts = hw_flat.value_counts().to_dict()
                        else:
                            hw_counts = {}

                        records.append({
                            "date":        str(snap_date),
                            "segments":    n_segs,
                            "total_km":    total_km,
                            "added":       added,
                            "removed":     removed,
                            "net_change":  net,
                            **{f"hw_{k}": v for k, v in hw_counts.items()},
                        })
                        prev_ids = cur_ids

                    except Exception as e:
                        st.warning(f"Snapshot {snap_date} failed: {e}")

                progress.empty()

                if not records:
                    st.error("No snapshots returned successfully.")
                else:
                    import pandas as pd
                    df = pd.DataFrame(records).fillna(0)

                    # Summary of metrics
                    st.divider()
                    sm1, sm2, sm3, sm4 = st.columns(4)
                    sm1.metric("Snapshots fetched",  len(df))
                    sm2.metric("Start network (km)", df["total_km"].iloc[0])
                    sm3.metric("End network (km)",   df["total_km"].iloc[-1])
                    net_total = round(df["total_km"].iloc[-1] - df["total_km"].iloc[0], 3)
                    sm4.metric("Total change (km)",  f"{net_total:+.3f}",
                               delta=f"{net_total:+.3f} km",
                               delta_color="normal" if net_total >= 0 else "inverse")

                    st.divider()

                    # Charts
                    chart_tab1, chart_tab2, chart_tab3 = st.tabs(
                        ["Total length over time", "Segments over time", "Added vs removed"]
                    )

                    with chart_tab1:
                        st.line_chart(df.set_index("date")["total_km"],
                                      use_container_width=True, height=280)
                        st.caption("Total road network length (km) at each snapshot")

                    with chart_tab2:
                        st.line_chart(df.set_index("date")["segments"],
                                      use_container_width=True, height=280)
                        st.caption("Number of road segments at each snapshot")

                    with chart_tab3:
                        st.bar_chart(df.set_index("date")[["added","removed"]],
                                     use_container_width=True, height=280)
                        st.caption("Road segments added (blue) and removed (red) at each interval")

                    st.divider()

                    # Full table
                    st.subheader("Snapshot table")
                    display_cols = ["date","segments","total_km","added","removed","net_change"]
                    display_cols = [c for c in display_cols if c in df.columns]

                    # Colour net_change column
                    def colour_net(val):
                        if val > 0:  return "color: #1D9E75; font-weight: 600"
                        if val < 0:  return "color: #E24B4A; font-weight: 600"
                        return ""

                    try:
                        styled = df[display_cols].style.map(colour_net, subset=["net_change"])
                    except AttributeError:
                        styled = df[display_cols].style.applymap(colour_net, subset=["net_change"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

                    # Road type breakdown over time
                    hw_cols = [c for c in df.columns if c.startswith("hw_")]
                    if hw_cols:
                        st.divider()
                        st.subheader("Road type composition over time")
                        hw_df = df[["date"] + hw_cols].set_index("date")
                        hw_df.columns = [c.replace("hw_","") for c in hw_df.columns]
                        st.area_chart(hw_df, use_container_width=True, height=280)
                        st.caption("Number of segments per road type at each snapshot")

                    # Export full timeline as CSV
                    st.divider()
                    csv_buf = io.StringIO()
                    df.to_csv(csv_buf, index=False)
                    st.download_button(
                        "⬇️ Download timeline as CSV",
                        data=csv_buf.getvalue().encode(),
                        file_name="road_timeline.csv",
                        mime="text/csv",
                    )

    # Filter & Select
    with tab_filter:
        col1, col2 = st.columns([1, 2])
        with col1:
            if "highway" in gdf.columns:
                hw_series      = gdf["highway"].apply(lambda x: x[0] if isinstance(x, list) else x)
                all_types      = sorted(hw_series.dropna().unique().tolist())
                selected_types = st.multiselect("Highway types", all_types, default=all_types)
                filtered       = gdf[hw_series.isin(selected_types)].copy()
            else:
                filtered = gdf.copy()
                st.info("No highway attribute found.")
            min_len  = st.slider("Min segment length (m)", 0, 500, 0, step=10)
            lens_flt = filtered.to_crs(epsg=3857).geometry.length
            filtered = filtered[lens_flt >= min_len]
            st.metric("Segments after filter", f"{len(filtered):,}")

        with col2:
            dcols = [c for c in ["name","highway","oneway","lanes","maxspeed","surface","geometry"]
                     if c in filtered.columns]
            st.dataframe(filtered[dcols].head(200), use_container_width=True, height=360)
            if len(filtered) > 200:
                st.caption(f"Showing first 200 of {len(filtered):,} rows")

        st.session_state.filtered_gdf = filtered

    # Statistics
    with tab_stats:
        work = st.session_state.get("filtered_gdf", gdf)
        lens = work.to_crs(epsg=3857).geometry.length
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Segments",     f"{len(work):,}")
        c2.metric("Total length", f"{lens.sum()/1000:.2f} km")
        c3.metric("Avg segment",  f"{lens.mean():.0f} m")
        c4.metric("Longest",      f"{lens.max():.0f} m")
        st.divider()
        if "highway" in work.columns:
            st.subheader("Length by road type")
            hw_flat = work["highway"].apply(lambda x: x[0] if isinstance(x, list) else x)
            tl = (lens.groupby(hw_flat).sum() / 1000).sort_values(ascending=False).reset_index()
            tl.columns = ["highway_type", "length_km"]
            tl["length_km"] = tl["length_km"].round(3)
            tl["share_%"]   = (tl["length_km"] / tl["length_km"].sum() * 100).round(1)
            st.dataframe(tl, use_container_width=True, hide_index=True)
            st.bar_chart(tl.set_index("highway_type")["length_km"])

    # Export
    with tab_export:
        export_gdf = st.session_state.get("filtered_gdf", gdf).copy()
        st.write(f"**{len(export_gdf):,}** segments ready  ·  date: **{p['date_label']}**")

        if not st.checkbox("Include all OSM attributes", value=False):
            keep = [c for c in ["name","highway","oneway","lanes","maxspeed","surface","geometry"]
                    if c in export_gdf.columns]
            export_gdf = export_gdf[keep]

        fmt = st.radio("Format", ["Shapefile (.shp)", "GeoJSON (.geojson)"], horizontal=True)

        if st.button("⬇️ Prepare download", type="primary"):
            if fmt == "GeoJSON (.geojson)":
                buf = io.BytesIO()
                export_gdf.to_crs(epsg=4326).to_file(buf, driver="GeoJSON")
                buf.seek(0)
                st.download_button("Download roads.geojson", data=buf,
                                   file_name="roads.geojson", mime="application/geo+json")
            else:
                buf = make_zip_shp(export_gdf, "roads")
                if buf:
                    st.download_button("Download roads_shapefile.zip", data=buf,
                                       file_name="roads_shapefile.zip", mime="application/zip")
                    st.caption("ZIP contains all 5 Shapefile components (.shp .dbf .shx .prj .cpg)")

    # Presets
    with tab_presets:
        st.subheader("📌 Preset coordinates")
        st.caption(
            "Manage saved locations for Sudan field work. "
            "Presets are stored in `presets.csv` next to this script — "
            "the file never leaves your machine."
        )

        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            template = (
                "# Sudan Road Extractor — Preset Coordinates\n"
                "# Required : lat, lon\n"
                "# Optional : date (YYYY-MM-DD) — leave blank for current OSM data\n"
                "# Maximum 50 rows.\n"
                "lat,lon,date\n"
            )
            st.download_button(
                "⬇️ Download blank template",
                data=template.encode(),
                file_name="presets.csv",
                mime="text/csv",
            )
        with col_info:
            st.info(
                f"**CSV location:** `{PRESET_FILE}`  \n"
                "Edit it in any spreadsheet app or text editor, then **restart the app** "
                "or press **R** in the terminal to reload."
            )

        st.divider()

        if presets:
            st.success(f"✅ {len(presets)} preset(s) loaded")

            import pandas as pd
            pdf = pd.DataFrame(presets)[["lat","lon","date"]]
            pdf["date"] = pdf["date"].fillna("current")
            st.dataframe(pdf, use_container_width=True, hide_index=True)

            st.subheader("All preset locations")
            pm = folium.Map(
                location=[sum(p["lat"] for p in presets)/len(presets),
                           sum(p["lon"] for p in presets)/len(presets)],
                zoom_start=6,
                tiles="CartoDB positron",
            )
            for i, preset in enumerate(presets):
                folium.Marker(
                    location=[preset["lat"], preset["lon"]],
                    tooltip=(
                        f"<b>#{i+1}</b><br>"
                        f"{preset['lat']:.5f}, {preset['lon']:.5f}"
                        + (f"<br>Date: {preset['date']}" if preset["date"] else "")
                    ),
                    icon=folium.Icon(color="purple", icon="map-marker", prefix="fa"),
                ).add_to(pm)
            st_folium(pm, width="100%", height=420, returned_objects=[])

        else:
            st.warning(
                f"No presets found at `{PRESET_FILE}`.  \n"
                "Download the blank template above, fill it in, save it as `presets.csv` "
                "in the same folder as this script, then reload the app."
            )
            st.markdown("""
**CSV format reference**

| Column | Required | Example |
|--------|----------|---------|
| `lat` | ✅ | `15.552700` |
| `lon` | ✅ | `32.532400` |
| `date` | ☐ | `2023-04-15` — leave blank for current OSM |
""")

else:
    st.info("👈 Enter coordinates in the sidebar and click **Fetch Roads** to get started.")
    with st.expander("ℹ️ How to use"):
        st.markdown("""
1. Enter **latitude/longitude** and **search radius** in the sidebar
2. Toggle **Query a specific date** to fetch roads as they were on any date back to Sept 2012
3. Click **Fetch Roads**
4. Use **📍 Nearest Road** to snap any coordinate to the closest road
5. Use **🕐 Compare Dates** to see what roads were added or removed between two dates
6. Use **📈 Time-Interval Analysis** to track road network changes across monthly/quarterly/yearly snapshots
7. Filter, view statistics, and export as Shapefile or GeoJSON
        """)