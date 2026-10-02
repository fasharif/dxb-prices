"""Streamlit page: value one Dubai apartment through the estimate API.

Run with ``streamlit run src/dxb_prices/ui/streamlit_app.py`` and point
``DXB_API_URL`` at the API (default http://127.0.0.1:8000).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dxb_prices.ui.client import ApiClient, ApiError

ROOM_CHOICES = ["studio", "1", "2", "3", "4", "5+", "penthouse"]
NO_PROJECT = "Not given"

st.set_page_config(page_title="Dubai apartment price estimate", layout="centered")
st.title("Dubai apartment price estimate")
st.caption(
    "A statistical estimate from a model trained on Dubai Land Department open transaction "
    "data. It is not a valuation. This project is not affiliated with DLD."
)

client = ApiClient()


@st.cache_data(ttl=300, show_spinner=False)
def load_communities(base_url: str) -> list[dict[str, object]]:
    return ApiClient(base_url=base_url).communities()


@st.cache_data(ttl=300, show_spinner=False)
def load_projects(base_url: str, community: str) -> list[str]:
    return [str(p["name"]) for p in ApiClient(base_url=base_url).projects(community)]


try:
    communities = load_communities(client.base_url)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

names = [str(c["name"]) for c in communities]
default_index = names.index("Business Bay") if "Business Bay" in names else 0

# Outside the form, so the project list follows the chosen community.
community = st.selectbox("Community (DLD area)", names, index=default_index)
try:
    projects = load_projects(client.base_url, community)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

with st.form("estimate"):
    project = st.selectbox(
        "Project or building",
        [NO_PROJECT, *projects],
        help=(
            "Type to search. With the project the estimate is much more accurate; without it "
            "the community-level model is used."
        ),
    )
    left, right = st.columns(2)
    size = left.number_input("Size (sqm)", min_value=18.0, max_value=3000.0, value=75.0, step=1.0)
    rooms = right.selectbox("Rooms", ROOM_CHOICES, index=1)
    status = left.radio("Status", ["Ready", "Off-plan"], horizontal=True)
    when = right.date_input("Date", value=date.today())
    submitted = st.form_submit_button("Estimate")

if submitted:
    payload = {
        "community": community,
        "size_sqm": float(size),
        "rooms": rooms,
        "off_plan": status == "Off-plan",
        "project": None if project == NO_PROJECT else project,
        "transaction_date": when.isoformat() if isinstance(when, date) else None,
    }
    try:
        result = client.estimate(payload)
    except ApiError as exc:
        st.error(str(exc))
        if exc.suggestions:
            st.info("Did you mean: " + ", ".join(exc.suggestions))
        st.stop()

    st.metric("Estimated price", f"AED {result['estimate_aed']:,}")
    rng = result["range_80_aed"]
    st.write(
        f"80% range: AED {rng['low']:,} to AED {rng['high']:,} "
        f"(about AED {result['estimate_per_sqm_aed']:,} per sqm)."
    )
    if result.get("community_median_per_sqm_aed"):
        st.write(
            f"Baseline for comparison: {result['community']} training-period median of "
            f"AED {result['community_median_per_sqm_aed']:,} per sqm."
        )
    for warning in result.get("warnings", []):
        st.warning(warning)
    st.subheader("Top factors")
    st.caption(
        f"Each factor moves the price per sqm up or down from a starting point of "
        f"AED {result['base_per_sqm_aed']:,} (SHAP values from the model). Together with the "
        "remaining factors they multiply up to the estimate."
    )
    rows = [
        {"Factor": f["label"], "Value": f["value"], "Effect": f"{f['effect_pct']:+.1f}%"}
        for f in result["top_factors"]
    ]
    rows.append(
        {
            "Factor": "All other factors",
            "Value": "",
            "Effect": f"{result['other_factors_effect_pct']:+.1f}%",
        }
    )
    st.table(pd.DataFrame(rows).set_index("Factor"))
    model = result["model"]
    which = (
        "the model that knows the project"
        if result["model_variant"] == "full"
        else ("the community-level model (no project)")
    )
    st.caption(
        f"Estimated by {which}. Model {model['version']}, trained on sales from "
        f"{', '.join(model['trained_on_months'])}."
    )
