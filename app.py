"""
Hispanic Heritage Historical Data Explorer
A Streamlit classroom app for World Bank historical data and function analysis.
"""
from __future__ import annotations

import io
import math
import re
import textwrap
from datetime import date, datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
API_ROOT = "https://api.worldbank.org/v2"
COUNTRIES = {
    "Mexico": "MX", "Brazil": "BR", "Argentina": "AR", "Colombia": "CO",
    "Chile": "CL", "Peru": "PE", "Venezuela": "VE", "Ecuador": "EC", "Uruguay": "UY",
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "unit": "years",
        "definition": "Life expectancy at birth indicates how many years a newborn would live if current age-specific mortality patterns persisted."
    },
    "Unemployment, national estimate": {
        "code": "SL.UEM.TOTL.NE.ZS", "unit": "% of labor force",
        "definition": "The share of the labor force without work but available for and seeking employment, using national estimates."
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "unit": "current US$ per person",
        "definition": "Gross domestic product divided by the midyear population, expressed in current US dollars. It is not average wealth or household income."
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "unit": "births per 1,000 people per year",
        "definition": "The annual number of live births per 1,000 people. This is a rate, not a total birth count."
    },
    "Net migration": {
        "code": "SM.POP.NETM", "unit": "people (net estimate)",
        "definition": "The net number of migrants during a period (immigrants minus emigrants). It is not an emigration count."
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "unit": "per 100,000 people",
        "definition": "Intentional homicide victims per 100,000 population."
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "unit": "% annual change",
        "definition": "Annual percentage change in the consumer price index."
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "unit": "% of GDP",
        "definition": "Central government debt as a percentage of gross domestic product."
    },
    "Total reserves including gold": {
        "code": "FI.RES.TOTL.CD", "unit": "current US$",
        "definition": "Total reserves including gold, valued in current US dollars."
    },
    "External balance on goods and services": {
        "code": "NE.RSB.GNFS.ZS", "unit": "% of GDP",
        "definition": "External balance on goods and services expressed as a percentage of GDP."
    },
}
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]
CONTEXTS = {
    "Business": ("Revenue", "time", "years", "US dollars", "A simplified business measure over time."),
    "Engineering": ("displacement", "time", "seconds", "meters", "A simplified displacement model."),
    "Medical": ("concentration", "time", "hours", "mg/L", "A simplified concentration model; not medical advice."),
    "Sports science": ("position", "time", "seconds", "meters", "A simplified athlete-motion model."),
    "Biology": ("population", "time", "days", "organisms", "A simplified population model."),
    "Bioengineering": ("signal", "time", "seconds", "arbitrary units", "A simplified biological-signal model."),
    "Economic forecasting": ("indicator", "time", "years", "index units", "A simplified economic indicator model."),
    "Weather": ("temperature", "time", "hours", "°C", "A simplified temperature model."),
    "Custom": ("output", "input", "input units", "output units", "Describe your own modeled relationship."),
}
SUGGESTED = {
    "Business": "a*x**2 + b*x + c",
    "Engineering": "a*sin(b*x) + c",
    "Medical": "a*exp(-b*x) + c",
    "Sports science": "a*x**2 + b*x + c",
    "Biology": "a*exp(b*x)",
    "Bioengineering": "a*sin(b*x) + c",
    "Economic forecasting": "a*x + b",
    "Weather": "a*sin(b*x) + c",
    "Custom": "a*x + b",
}
PARAMETER_DEFAULTS = {"a": 1.0, "b": 0.2, "c": 0.0}

st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 " + APP_TITLE)
st.caption("Hispanic Heritage Data Lab • historical data, mathematical modeling, and classroom-friendly interpretation")
st.info("Brazil is included as a Latin American comparison country. Country inclusion does not imply that all residents share one identity or heritage.")

def wb_url(country_code: str, indicator_code: str, start: int, end: int) -> str:
    return f"{API_ROOT}/country/{country_code}/indicator/{indicator_code}?date={start}:{end}&format=json&per_page=200"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start: int, end: int):
    url = wb_url(country_code, indicator_code, start, end)
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "HispanicHeritageDataExplorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), url, "The API returned an empty or unexpected response."
        rows = []
        for item in payload[1]:
            year, value = item.get("date"), item.get("value")
            try:
                year = int(year)
            except (TypeError, ValueError):
                continue
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                rows.append({"year": year, "value": value})
        df = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year").reset_index(drop=True)
        return df, url, None if not df.empty else "The API returned no numeric observations for this period."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), url, "The World Bank request timed out. Try again later."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), url, f"World Bank request failed: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), url, f"Could not read the API response: {exc}"

def validate_data(df: pd.DataFrame, indicator_name: str):
    errors = []
    if df.empty:
        return ["There are no rows to validate."]
    if not {"year", "value"}.issubset(df.columns):
        return ["The table must contain year and value columns."]
    years = pd.to_numeric(df["year"], errors="coerce")
    values = pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Every year must be a whole number.")
    if years.duplicated().any():
        errors.append("Years must be unique; remove duplicate years.")
    if values.isna().any() or not np.isfinite(values.dropna()).all():
        errors.append("Every value must be a finite number.")
    if indicator_name in ("Life expectancy at birth", "Crude birth rate", "Intentional homicide rate", "Total reserves including gold") and (values.dropna() < 0).any():
        errors.append(f"{indicator_name} values cannot be negative.")
    if indicator_name == "Unemployment, national estimate" and ((values < 0) | (values > 100)).any():
        errors.append("Unemployment percentages must be between 0 and 100.")
    if indicator_name == "Central government debt" and (values.dropna() < 0).any():
        errors.append("Central government debt percentages cannot be negative.")
    if indicator_name == "Life expectancy at birth" and (values.dropna() > 125).any():
        errors.append("Life expectancy above 125 years is implausible; check the edited value.")
    return errors

def fit_polynomial(df: pd.DataFrame, degree: int):
    clean = df[["year", "value"]].copy()
    clean["year"] = pd.to_numeric(clean["year"], errors="coerce")
    clean["value"] = pd.to_numeric(clean["value"], errors="coerce")
    clean = clean.dropna().sort_values("year")
    if clean["year"].duplicated().any():
        raise ValueError("Years must be unique before fitting.")
    if len(clean) < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; found {len(clean)}.")
    base_year = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base_year
    y = clean["value"].to_numpy(dtype=float)
    coeffs = np.polyfit(t, y, degree)
    pred = np.polyval(coeffs, t)
    return {"coeffs": coeffs, "base_year": base_year, "degree": degree, "years": clean["year"].to_numpy(dtype=int),
            "values": y, "fitted": pred, "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
            "mae": float(mean_absolute_error(y, pred)), "mean": float(np.mean(y)), "n": len(y),
            "min_year": int(clean["year"].min()), "max_year": int(clean["year"].max()), "data": clean}

def poly_predict(model, years):
    arr = np.asarray(years, dtype=float)
    return np.polyval(model["coeffs"], arr - model["base_year"])

def poly_equation(model, value_name="y"):
    terms = []
    d = model["degree"]
    for i, coefficient in enumerate(model["coeffs"]):
        power = d - i
        c = f"{coefficient:.8g}"
        if power == 0:
            terms.append(c)
        elif power == 1:
            terms.append(f"({c})t")
        else:
            terms.append(f"({c})t^{power}")
    return f"{value_name}(t) = " + " + ".join(terms) + f", where t = year − {model['base_year']}"

def chronological_holdout(df, degree):
    clean = df[["year", "value"]].dropna().sort_values("year")
    # Keep the latest 20%, at least one point, aside only when enough points remain for fitting.
    test_n = max(1, int(round(len(clean) * 0.2)))
    train = clean.iloc[:-test_n]
    test = clean.iloc[-test_n:]
    if len(train) < degree + 1 or len(test) < 2:
        return None
    try:
        m = fit_polynomial(train, degree)
        pred = poly_predict(m, test["year"].to_numpy())
        return {"n": len(test), "mae": float(mean_absolute_error(test["value"], pred)),
                "r2": float(r2_score(test["value"], pred)) if len(test) > 1 else float("nan")}
    except (ValueError, np.linalg.LinAlgError):
        return None

def safe_expression(expr_text: str, x_symbol):
    allowed = {"x": x_symbol, "a": sp.Symbol("a"), "b": sp.Symbol("b"), "c": sp.Symbol("c"),
               "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
               "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E}
    if len(expr_text) > 200:
        raise ValueError("Expression is too long.")
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,^ \t]+", expr_text):
        raise ValueError("Use numbers, x, a, b, c, parentheses, arithmetic, and supported functions only.")
    expr_text = expr_text.replace("^", "**")
    try:
        expr = sp.sympify(expr_text, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse the expression. Example: a*x**2 + b*x + c.") from exc
    symbols = expr.free_symbols
    permitted_symbols = {x_symbol, sp.Symbol("a"), sp.Symbol("b"), sp.Symbol("c")}
    if not symbols.issubset(permitted_symbols):
        raise ValueError("Only x and parameters a, b, c are supported.")
    return expr

def fmt_num(x):
    try:
        return f"{float(x):,.6g}"
    except (ValueError, TypeError):
        return str(x)

def make_chart(df, model, query_years=None, query_values=None, comparison=None, sample_increment=1, title="Historical data and fitted curve", y_label="Value", horizon=0, printer_mode=False):
    fig, ax = plt.subplots(figsize=(8.5, 4.2) if printer_mode else (10, 5.2))
    years = df["year"].to_numpy(dtype=float)
    vals = df["value"].to_numpy(dtype=float)
    ax.scatter(years, vals, label="Observed / edited data", s=26, zorder=3)
    display_end = max(model["max_year"], model["max_year"] + int(horizon))
    if query_years is not None and len(query_years):
        display_end = max(display_end, float(np.max(query_years)))
    display_end = min(display_end, 2100)
    historical_x = np.linspace(model["min_year"], max(model["max_year"], model["min_year"] + 1), 400)
    ax.plot(historical_x, poly_predict(model, historical_x), label="Polynomial fit (historical range)", linewidth=2)
    if display_end > model["max_year"]:
        future_x = np.linspace(model["max_year"], display_end, 250)
        ax.plot(future_x, poly_predict(model, future_x), linestyle="--", linewidth=2, label="Extrapolated model curve")
    ax.axvline(model["max_year"], linestyle=":", linewidth=1.2, label="Historical boundary")
    if query_years is not None and len(query_years):
        qy = np.asarray(query_years, dtype=float)
        qv = np.asarray(query_values, dtype=float)
        outside = (qy < model["min_year"]) | (qy > model["max_year"])
        if outside.any():
            ax.plot(qy[outside], qv[outside], linestyle="None", marker="x", markersize=8, label="Extrapolated estimate")
        if (~outside).any():
            ax.plot(qy[~outside], qv[~outside], linestyle="None", marker="D", markersize=5, label="Calculated estimate (within range)")
    sample_years = np.arange(model["min_year"], model["max_year"] + 1, sample_increment)
    ax.plot(sample_years, poly_predict(model, sample_years), linestyle="None", marker=".", markersize=4, label=f"Model samples every {sample_increment} year(s)")
    if comparison is not None and not comparison.empty:
        ax.plot(comparison["year"], comparison["value"], marker="o", markersize=3, linewidth=1, label="Comparison country")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def create_pdf(summary, chart_bytes, data_preview, calc_results):
    buff = io.BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch,
                            topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = [Paragraph(APP_TITLE, styles["Title"]), Spacer(1, 8)]
    for line in summary:
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["Small"]))
    story.append(Spacer(1, 8))
    chart_buff = io.BytesIO(chart_bytes)
    story.append(RLImage(chart_buff, width=9.5*inch, height=4.5*inch))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Data preview", styles["Heading2"]))
    rows = [list(data_preview.columns)] + data_preview.astype(str).values.tolist()
    table = Table(rows[:26], repeatRows=1)
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),
                               ("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(table)
    if calc_results is not None and not calc_results.empty:
        story += [Spacer(1, 8), Paragraph("Calculated-year results", styles["Heading2"])]
        rows = [list(calc_results.columns)] + calc_results.astype(str).values.tolist()
        table = Table(rows[:26], repeatRows=1)
        table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),
                                   ("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(table)
    doc.build(story)
    return buff.getvalue()

st.sidebar.header("Data and model controls")
country_name = st.sidebar.selectbox("Country", list(COUNTRIES))
indicator_name = st.sidebar.selectbox("Indicator", list(INDICATORS))
degree = st.sidebar.slider("Polynomial degree", 3, 8, 3)
sample_increment = st.sidebar.slider("Graph sample increment (years)", 1, 10, 2)
horizon = st.sidebar.slider("Forward display horizon (years)", 0, 50, 10)
comparison_name = st.sidebar.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_name])
printer_mode = st.sidebar.checkbox("Printer-friendly report view", value=False)
current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
country_code = COUNTRIES[country_name]
indicator = INDICATORS[indicator_name]

with st.spinner("Retrieving historical data from the World Bank API..."):
    source_df, source_url, api_message = fetch_world_bank(country_code, indicator["code"], start_year, end_year)
if api_message:
    st.warning(api_message)
st.caption(f"Requested years: {start_year}–{end_year}. Retrieved on {date.today().isoformat()}.")
st.markdown(f"**Indicator definition:** {indicator['definition']}  \n**Unit:** {indicator['unit']}  \n**World Bank indicator:** `{indicator['code']}` • [API source]({source_url})")
if source_df.empty:
    st.error("No source observations are available. Check your connection or try again later. The app will not substitute fabricated values.")
    source_df = pd.DataFrame(columns=["year", "value"])

if "source_signature" not in st.session_state:
    st.session_state.source_signature = None
signature = (country_code, indicator["code"], start_year, end_year)
if st.session_state.source_signature != signature:
    st.session_state.source_signature = signature
    st.session_state.working_data = source_df.copy()
    st.session_state.user_modified = False
    st.session_state.single_result = None
    st.session_state.batch_result = None

left, right = st.columns([3, 1])
with left:
    st.subheader("Validated editable data")
    st.caption("Edit year/value rows below. Source data remains unchanged; edits create a user-modified scenario.")
    edited = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True,
                            column_config={"year": st.column_config.NumberColumn("Year", step=1, format="%d"),
                                           "value": st.column_config.NumberColumn(f"Value ({indicator['unit']})", format="%.6g")})
    if not edited.equals(st.session_state.working_data):
        st.session_state.working_data = edited.copy()
        st.session_state.user_modified = True
        st.session_state.single_result = None
        st.session_state.batch_result = None
with right:
    st.subheader("Data status")
    st.metric("Numeric observations", len(st.session_state.working_data))
    st.metric("Actual coverage", f"{int(st.session_state.working_data.year.min())}–{int(st.session_state.working_data.year.max())}" if not st.session_state.working_data.empty and pd.to_numeric(st.session_state.working_data.year, errors="coerce").notna().any() else "None")
    st.write("**Scenario:** " + ("User-modified" if st.session_state.user_modified else "Source data"))
    if st.button("Reset to source", use_container_width=True):
        st.session_state.working_data = source_df.copy()
        st.session_state.user_modified = False
        st.session_state.single_result = None
        st.session_state.batch_result = None
        st.rerun()

validation_errors = validate_data(st.session_state.working_data, indicator_name)
for err in validation_errors:
    st.error(err)

model = None
if not validation_errors:
    try:
        model = fit_polynomial(st.session_state.working_data, degree)
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))

if model is not None:
    st.subheader("Polynomial fit")
    if printer_mode:
        st.caption("Printer-friendly view: concise metrics and a simplified chart are emphasized for printing or saving as PDF.")
    st.latex(poly_equation(model, indicator_name))
    metrics = st.columns(5)
    metrics[0].metric("Degree", model["degree"])
    metrics[1].metric("Observations", model["n"])
    metrics[2].metric("In-sample R²", fmt_num(model["r2"]))
    metrics[3].metric("In-sample MAE", fmt_num(model["mae"]))
    metrics[4].metric("Mean", fmt_num(model["mean"]))
    st.caption(f"Centered time: t = year − {model['base_year']}. Historical fit range: {model['min_year']}–{model['max_year']}. Metrics describe the fitted data and are not proof of future accuracy.")
    holdout = chronological_holdout(st.session_state.working_data, degree)
    if holdout:
        st.write(f"**Chronological holdout (separate from in-sample metrics):** {holdout['n']} later observations; MAE = {holdout['mae']:.6g}; R² = {holdout['r2']:.6g}.")
    else:
        st.caption("Chronological holdout score not shown: there are not enough observations to leave a useful later test segment and still fit this degree.")

    st.subheader("Calculate a single year")
    c1, c2 = st.columns([1, 2])
    with c1:
        calc_year = st.number_input("Year (1950–2100)", min_value=1950, max_value=2100, value=min(max(current_year, 1950), 2100), step=1)
        if st.button("Calculate value", type="primary"):
            value = float(poly_predict(model, [calc_year])[0])
            in_range = model["min_year"] <= calc_year <= model["max_year"]
            observed_match = st.session_state.working_data[pd.to_numeric(st.session_state.working_data.year, errors="coerce") == calc_year]
            st.session_state.single_result = {"year": int(calc_year), "estimate": value, "classification": "Within fitted range" if in_range else "Outside fitted range (extrapolation)",
                                              "observed": float(observed_match.iloc[0]["value"]) if not observed_match.empty else None,
                                              "signature": (country_code, indicator["code"], degree, tuple(st.session_state.working_data.astype(str).itertuples(index=False, name=None)))}
    with c2:
        result = st.session_state.single_result
        if result:
            st.metric(f"Model estimate for {result['year']}", f"{result['estimate']:,.6g} {indicator['unit']}")
            st.write(result["classification"])
            if result["observed"] is not None:
                st.write(f"Actual edited/source observation for that year: **{result['observed']:,.6g}**. The fitted estimate is not the actual observation.")
            if result["year"] < model["min_year"] - 10 or result["year"] > model["max_year"] + 10:
                st.warning("This is a distant extrapolation. Polynomial curves can grow or turn sharply outside the observed years; do not treat it as a forecast guarantee.")

    st.subheader("Batch year calculations")
    mode = st.radio("Choose batch input", ["Year range", "Custom comma-separated years"], horizontal=True)
    batch_years = []
    if mode == "Year range":
        bc1, bc2, bc3 = st.columns(3)
        with bc1: batch_start = st.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, model["min_year"]), key="batch_start")
        with bc2: batch_end = st.number_input("End year", min_value=1950, max_value=2100, value=min(2100, model["max_year"]), key="batch_end")
        with bc3: batch_step = st.slider("Range step", 1, 10, 1)
        if batch_start <= batch_end:
            batch_years = list(range(int(batch_start), int(batch_end) + 1, int(batch_step)))
        else:
            st.error("Start year must be less than or equal to end year.")
    else:
        custom_year_text = st.text_input("Years separated by commas", "2000, 2010, 2020")
        try:
            pieces = [p.strip() for p in custom_year_text.split(",") if p.strip()]
            parsed = [int(p) for p in pieces]
            if any(str(y) != p and not re.fullmatch(r"[+-]?\d+", p) for y, p in zip(parsed, pieces)):
                raise ValueError
            batch_years = list(dict.fromkeys(parsed))
            invalid = [y for y in batch_years if y < 1950 or y > 2100]
            if invalid:
                st.error(f"Years must be between 1950 and 2100. Invalid: {invalid}")
                batch_years = []
        except ValueError:
            st.error("Enter whole years separated by commas, such as 1990, 2000, 2010.")
            batch_years = []
    if st.button("Calculate batch results"):
        if not batch_years:
            st.error("Enter at least one valid year.")
        else:
            bvals = poly_predict(model, batch_years)
            rows = []
            data = st.session_state.working_data.copy()
            data["year"] = pd.to_numeric(data.year, errors="coerce")
            for y, v in zip(batch_years, bvals):
                actual = data.loc[data.year == y, "value"]
                rows.append({"year": int(y), "estimate": float(v), "unit": indicator["unit"],
                             "classification": "Within fitted range" if model["min_year"] <= y <= model["max_year"] else "Outside fitted range (extrapolation)",
                             "actual observation": float(actual.iloc[0]) if len(actual) else np.nan})
            st.session_state.batch_result = pd.DataFrame(rows)
    if st.session_state.batch_result is not None:
        st.dataframe(st.session_state.batch_result, use_container_width=True)
        far = st.session_state.batch_result["classification"].str.startswith("Outside")
        if far.any():
            st.warning("Batch results include extrapolations. They are model estimates, not validated predictions or 95% prediction intervals.")

    comparison_df = pd.DataFrame()
    if comparison_name != "None":
        comp, comp_url, comp_msg = fetch_world_bank(COUNTRIES[comparison_name], indicator["code"], start_year, end_year)
        if comp_msg:
            st.warning(f"Comparison-country data: {comp_msg}")
        if not comp.empty:
            comparison_df = comp
            st.caption(f"Comparison source: [{comparison_name} World Bank API]({comp_url}). Coverage can differ; gaps mean the API did not provide numeric observations for those years.")
    query_years, query_values = None, None
    if st.session_state.batch_result is not None:
        query_years = st.session_state.batch_result["year"].to_numpy()
        query_values = st.session_state.batch_result["estimate"].to_numpy()
    elif st.session_state.single_result:
        query_years = np.array([st.session_state.single_result["year"]])
        query_values = np.array([st.session_state.single_result["estimate"]])
    fig = make_chart(st.session_state.working_data.assign(year=pd.to_numeric(st.session_state.working_data.year), value=pd.to_numeric(st.session_state.working_data.value)),
                     model, query_years, query_values, comparison_df if not comparison_df.empty else None,
                     sample_increment, f"{country_name}: {indicator_name}", indicator["unit"], horizon=horizon, printer_mode=printer_mode)
    st.subheader("Visualization")
    st.pyplot(fig, use_container_width=True)
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", dpi=160, bbox_inches="tight")
    chart_bytes = img_buf.getvalue()
    plt.close(fig)

    st.subheader("Interpretation by mathematics pathway")
    st.write(f"**Historical trend:** the model is a degree-{degree} polynomial fitted to {model['n']} numeric observations. Its curve summarizes a pattern in the selected data; it does not establish why the pattern occurred.")
    st.write(f"**Average rate of change:** over any two years a and b with a ≠ b, calculate (f(b) − f(a)) / (b − a). Its unit is {indicator['unit']} per year.")
    if degree >= 2:
        derivative_coeffs = np.polyder(model["coeffs"])
        second_coeffs = np.polyder(model["coeffs"], 2)
        st.write(f"**Degree and shape:** changing the degree can change curvature and turning behavior. The first derivative of the fitted polynomial is `{np.array2string(derivative_coeffs, precision=5)}` in centered-time coordinates; the second derivative is `{np.array2string(second_coeffs, precision=5)}`.")
    if "Precalculus" in COURSES and False:
        pass
    course = st.selectbox("Mathematics pathway for interpretation", COURSES, index=2, key="course")
    if course == "Honors Algebra 2":
        st.write("Focus: domain and range of the displayed model, end behavior, graphical trends, and average rates of change. A polynomial's mathematical domain is all real numbers, but the historical data domain is only the observed years.")
    elif course == "Precalculus":
        st.write("Focus: coefficient and degree effects, turning points, and concavity. Turning behavior can be explored by checking where the first derivative changes sign; concavity is related to the second derivative.")
    else:
        st.write("Focus: derivatives, tangent/secant lines, critical points, extrema, and inflection points. The historical polynomial analysis below is numerical and does not imply that the real-world process follows a polynomial outside the data range.")

    # Course-independent context/function explorer
    st.header("Function Lab: contexts and calculus")
    context = st.selectbox("Context", list(CONTEXTS))
    default_dep, default_ind, default_ind_unit, default_dep_unit, default_desc = CONTEXTS[context]
    description = st.text_input("Context description", default_desc)
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        dep_name = st.text_input("Dependent variable", default_dep, key="dep_name")
        dep_unit = st.text_input("Dependent-variable units", default_dep_unit, key="dep_unit")
    with vcol2:
        ind_name = st.text_input("Independent variable", default_ind, key="ind_name")
        ind_unit = st.text_input("Independent-variable units", default_ind_unit, key="ind_unit")
    function_source = st.radio("Function source", ["Use historical curve of best fit", "Suggested function", "Build from parameters", "Enter my own function"], horizontal=False)
    a_val = st.number_input("Parameter a", value=1.0, step=0.1)
    b_val = st.number_input("Parameter b", value=0.2, step=0.1)
    c_val = st.number_input("Parameter c", value=0.0, step=0.1)
    if function_source == "Use historical curve of best fit":
        function_text = "Historical fitted polynomial"
    elif function_source == "Suggested function":
        function_text = SUGGESTED[context]
    elif function_source == "Build from parameters":
        function_text = SUGGESTED[context]
    else:
        function_text = st.text_input("Function in x (examples: x**2, a*x+b, sin(x), exp(-x))", "a*x**2 + b*x + c")
    x = sp.Symbol("x", real=True)
    symbolic = function_source != "Use historical curve of best fit"
    expr = None
    parse_error = None
    if symbolic:
        try:
            expr = safe_expression(function_text, x).subs({sp.Symbol("a"): a_val, sp.Symbol("b"): b_val, sp.Symbol("c"): c_val})
            expr = sp.simplify(expr)
            fn = sp.lambdify(x, expr, modules=["numpy"])
            first = sp.diff(expr, x)
            second = sp.diff(first, x)
            first_fn = sp.lambdify(x, first, modules=["numpy"])
            second_fn = sp.lambdify(x, second, modules=["numpy"])
            st.latex(sp.latex(sp.Eq(sp.Symbol(dep_name), expr)))
            st.write(f"**First derivative:**")
            st.latex(sp.latex(first))
            st.write("**Second derivative:**")
            st.latex(sp.latex(second))
        except Exception as exc:
            parse_error = str(exc)
            st.error(f"Function input error: {parse_error}")
    else:
        if model is not None:
            # Transform to calendar-year input for consistent predictions.
            expr = sum(sp.Float(float(coef)) * (x - model["base_year"])**(model["degree"]-i)
                       for i, coef in enumerate(model["coeffs"]))
            fn = sp.lambdify(x, expr, modules=["numpy"])
            first, second = sp.diff(expr, x), sp.diff(expr, x, 2)
            first_fn, second_fn = sp.lambdify(x, first, modules=["numpy"]), sp.lambdify(x, second, modules=["numpy"])
            st.latex(sp.latex(sp.Eq(sp.Symbol(dep_name), expr)))
            st.caption("For this option, x represents calendar year; the polynomial itself uses centered time internally.")
            st.write("**First derivative:**"); st.latex(sp.latex(first))
            st.write("**Second derivative:**"); st.latex(sp.latex(second))
        else:
            parse_error = "A valid historical fit is required."
    if expr is not None and parse_error is None:
        lo = float(st.number_input("Analysis interval start", value=float(model["min_year"] if function_source == "Use historical curve of best fit" and model else 0.0), key="interval_lo"))
        hi = float(st.number_input("Analysis interval end", value=float(model["max_year"] if function_source == "Use historical curve of best fit" and model else 10.0), key="interval_hi"))
        if hi <= lo:
            st.error("The interval end must be greater than the start.")
        else:
            x0 = st.slider("Current input value", min_value=float(lo), max_value=float(hi), value=float((lo+hi)/2), key="moving_x")
            s1, s2 = st.columns(2)
            with s1: show_f = st.checkbox("Show function", True)
            with s2: show_d = st.checkbox("Show first derivative", True)
            show_dd = st.checkbox("Show second derivative", False)
            x1 = st.slider("Secant interval start", min_value=float(lo), max_value=float(hi), value=float(lo), key="secant_a")
            x2 = st.slider("Secant interval end", min_value=float(lo), max_value=float(hi), value=float(hi), key="secant_b")
            xx = np.linspace(lo, hi, 700)
            def safe_eval(func, vals):
                with np.errstate(all="ignore"):
                    out = np.asarray(func(vals), dtype=float)
                if out.shape == ():
                    out = np.full_like(vals, float(out), dtype=float)
                out[~np.isfinite(out)] = np.nan
                return out
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            if show_f:
                ax2.plot(xx, safe_eval(fn, xx), label=f"{dep_name} = f({ind_name})")
            if show_d:
                ax2.plot(xx, safe_eval(first_fn, xx), label="First derivative")
            if show_dd:
                ax2.plot(xx, safe_eval(second_fn, xx), label="Second derivative")
            y0 = float(safe_eval(fn, np.array([x0]))[0])
            dy0 = float(safe_eval(first_fn, np.array([x0]))[0])
            if np.isfinite(y0):
                ax2.plot([x0], [y0], marker="o", markersize=7, label="Moving point")
                if np.isfinite(dy0):
                    tangent_x = np.linspace(max(lo, x0-(hi-lo)*0.12), min(hi, x0+(hi-lo)*0.12), 80)
                    ax2.plot(tangent_x, y0 + dy0*(tangent_x-x0), linestyle="--", label="Tangent line")
            y1 = float(safe_eval(fn, np.array([x1]))[0])
            y2 = float(safe_eval(fn, np.array([x2]))[0])
            if x2 != x1 and np.isfinite(y1) and np.isfinite(y2):
                secant_slope = (y2-y1)/(x2-x1)
                secant_x = np.array([x1, x2])
                ax2.plot(secant_x, np.array([y1, y2]), linestyle=":", linewidth=2, label="Secant line")
            ax2.set_xlabel(f"{ind_name} ({ind_unit})")
            ax2.set_ylabel(f"{dep_name} ({dep_unit})")
            ax2.grid(True, alpha=0.25)
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            st.write(f"**At {ind_name} = {x0:g} {ind_unit}:** f(x) ≈ {fmt_num(y0)} {dep_unit}; f′(x) ≈ {fmt_num(dy0)} {dep_unit}/{ind_unit}.")
            if np.isfinite(float(safe_eval(second_fn, np.array([x0]))[0])):
                st.write(f"f″(x) ≈ {fmt_num(float(safe_eval(second_fn, np.array([x0]))[0]))} {dep_unit}/{ind_unit}². The first derivative describes instantaneous rate; the second describes how that rate changes.")
            if x2 != x1 and np.isfinite(y1) and np.isfinite(y2):
                st.write(f"**Average rate of change** from {x1:g} to {x2:g}: {fmt_num((y2-y1)/(x2-x1))} {dep_unit}/{ind_unit}.")
            st.write(f"**Context:** {description} The model output is measured in {dep_unit}; its rate of change is measured in {dep_unit} per {ind_unit}.")
            # Numerical critical points and inflection points, checked by sign change.
            if course == "Calculus":
                st.subheader("Numerical derivative analysis")
                try:
                    d_expr = sp.lambdify(x, first, modules=["numpy"])
                    dd_expr = sp.lambdify(x, second, modules=["numpy"])
                    crit_candidates = []
                    for root in sp.solve(sp.Eq(first, 0), x):
                        if root.is_real:
                            rv = float(root)
                            if lo <= rv <= hi and np.isfinite(float(safe_eval(fn, np.array([rv]))[0])):
                                crit_candidates.append(rv)
                    crit_rows = []
                    for rv in sorted(set(round(v, 8) for v in crit_candidates)):
                        eps = max((hi-lo)*1e-5, 1e-6)
                        left_d = float(safe_eval(first_fn, np.array([max(lo, rv-eps)]))[0])
                        right_d = float(safe_eval(first_fn, np.array([min(hi, rv+eps)]))[0])
                        if left_d < 0 < right_d: kind = "Local minimum (derivative changes − to +)"
                        elif left_d > 0 > right_d: kind = "Local maximum (derivative changes + to −)"
                        else: kind = "Critical point; no sign-change extremum confirmed"
                        crit_rows.append({"x": rv, "f(x)": float(safe_eval(fn, np.array([rv]))[0]), "classification": kind})
                    st.write("Critical points (numerical/symbolic candidates; classifications checked with nearby derivative signs):")
                    st.dataframe(pd.DataFrame(crit_rows) if crit_rows else pd.DataFrame(columns=["x", "f(x)", "classification"]), use_container_width=True)
                    grid = np.linspace(lo, hi, 800)
                    dv = safe_eval(first_fn, grid)
                    intervals = []
                    start_i = 0
                    for i in range(1, len(grid)):
                        if np.isfinite(dv[i-1:i+1]).all() and np.sign(dv[i]) != np.sign(dv[i-1]):
                            intervals.append((grid[start_i], grid[i-1], "increasing" if np.nanmean(dv[start_i:i]) > 0 else "decreasing"))
                            start_i = i
                    intervals.append((grid[start_i], grid[-1], "increasing" if np.nanmean(dv[start_i:]) > 0 else "decreasing"))
                    st.write("Approximate increasing/decreasing intervals:")
                    st.dataframe(pd.DataFrame(intervals, columns=["start", "end", "behavior"]))
                    infl = []
                    for root in sp.solve(sp.Eq(second, 0), x):
                        if root.is_real:
                            rv = float(root)
                            if lo < rv < hi:
                                eps = max((hi-lo)*1e-5, 1e-6)
                                lval = float(safe_eval(second_fn, np.array([rv-eps]))[0])
                                rval = float(safe_eval(second_fn, np.array([rv+eps]))[0])
                                if np.isfinite(lval) and np.isfinite(rval) and np.sign(lval) != np.sign(rval):
                                    infl.append({"x": rv, "f(x)": float(safe_eval(fn, np.array([rv]))[0]), "concavity changes": True})
                    st.write("Inflection points are included only when the second derivative changes sign:")
                    st.dataframe(pd.DataFrame(infl) if infl else pd.DataFrame(columns=["x", "f(x)", "concavity changes"]), use_container_width=True)
                    vals_at_ends = [(lo, float(safe_eval(fn, np.array([lo]))[0])), (hi, float(safe_eval(fn, np.array([hi]))[0]))]
                    vals_at_ends += [(r, float(safe_eval(fn, np.array([r]))[0])) for r in crit_candidates]
                    finite_vals = [(xxv, yyv) for xxv, yyv in vals_at_ends if np.isfinite(yyv)]
                    if finite_vals:
                        st.write(f"**Closed-interval extrema (candidates include endpoints):** min ≈ {min(finite_vals, key=lambda z:z[1])}; max ≈ {max(finite_vals, key=lambda z:z[1])}.")
                    st.caption("Results are numerical approximations where applicable. For rational/logarithmic functions, check domain restrictions and discontinuities; graph gaps are not joined across non-finite values.")
                except Exception as exc:
                    st.warning(f"Derivative analysis could not be completed for this expression: {exc}")
            plt.close(fig2)

    # Export artifacts
    st.header("Export current work")
    summary_lines = [
        f"Country: {country_name} ({country_code}) | Indicator: {indicator_name} ({indicator['code']})",
        f"Unit: {indicator['unit']} | Source retrieval date: {date.today().isoformat()} | API: {source_url}",
        f"Data status: {'User-modified scenario' if st.session_state.user_modified else 'Source data'}",
        f"Coverage: {model['min_year']}–{model['max_year']} | n={model['n']} | Degree={degree}",
        f"Equation: {poly_equation(model, indicator_name)}",
        f"In-sample R²={model['r2']:.8g}; in-sample MAE={model['mae']:.8g}; mean={model['mean']:.8g}",
        "Limitations: polynomial fits can be sensitive to degree and extrapolation; MAE/residuals describe in-sample error and are not prediction intervals. Do not infer historical causes from this curve.",
    ]
    export_df = st.session_state.working_data.copy()
    export_df["year"] = pd.to_numeric(export_df["year"], errors="coerce")
    export_df["value"] = pd.to_numeric(export_df["value"], errors="coerce")
    residuals = model["values"] - model["fitted"]
    residual_df = pd.DataFrame({"year": model["years"], "actual": model["values"], "fitted": model["fitted"], "residual": residuals})
    results_df = st.session_state.batch_result if st.session_state.batch_result is not None else pd.DataFrame(
        [{"year": st.session_state.single_result["year"], "estimate": st.session_state.single_result["estimate"],
          "unit": indicator["unit"], "classification": st.session_state.single_result["classification"],
          "actual observation": st.session_state.single_result["observed"]}] if st.session_state.single_result else [])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    xlsx_buff = io.BytesIO()
    with pd.ExcelWriter(xlsx_buff, engine="xlsxwriter") as writer:
        pd.DataFrame({"item": ["Country", "Country code", "Indicator", "Indicator code", "Unit", "Retrieval date", "Data status", "Equation", "Degree", "Base year", "R-squared (in-sample)", "MAE (in-sample)", "Mean", "Source URL"],
                      "value": [country_name, country_code, indicator_name, indicator["code"], indicator["unit"], date.today().isoformat(),
                                "User-modified scenario" if st.session_state.user_modified else "Source data", poly_equation(model, indicator_name), degree,
                                model["base_year"], model["r2"], model["mae"], model["mean"], source_url]}).to_excel(writer, sheet_name="Summary", index=False)
        export_df.to_excel(writer, sheet_name="Working data", index=False)
        residual_df.to_excel(writer, sheet_name="Fit and residuals", index=False)
        results_df.to_excel(writer, sheet_name="Calculated years", index=False)
    pdf_bytes = create_pdf(summary_lines, chart_bytes, export_df.head(25), results_df)
    d1, d2, d3 = st.columns(3)
    with d1: st.download_button("Download CSV", csv_bytes, file_name="historical_working_data.csv", mime="text/csv", use_container_width=True)
    with d2: st.download_button("Download Excel (.xlsx)", xlsx_buff.getvalue(), file_name="historical_data_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with d3: st.download_button("Download PDF report", pdf_bytes, file_name="historical_data_explorer_report.pdf", mime="application/pdf", use_container_width=True)
else:
    st.info("Choose a lower polynomial degree or provide enough distinct, valid observations to enable fitting and model exports.")

st.sidebar.markdown("---")
st.sidebar.caption("Data source: World Bank API. No API key is required. Results are educational models, not official forecasts.")
