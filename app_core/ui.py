"""Small, accessible presentation helpers for MacroMate’s guided meal flow."""

import base64
from datetime import date
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ICONS = {
    "scan": '<rect x="6" y="6" width="12" height="12" rx="6"/><path d="M8 2H4a2 2 0 0 0-2 2v4m14-6h4a2 2 0 0 1 2 2v4M2 16v4a2 2 0 0 0 2 2h4m8 0h4a2 2 0 0 0 2-2v-4"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
    "shield": '<path d="m12 2 8 3v6c0 5-4 9-8 11-4-2-8-6-8-11V5l8-3Z"/><path d="m8 12 3 3 5-6"/>',
}


def icon(symbol, size=24):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[symbol]}</svg>')


def name(label):
    return label.replace("_", " ").title()


@lru_cache(maxsize=12)
def asset_uri(path):
    path = Path(path)
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def navigate(page):
    st.session_state.workspace = page


def setup():
    st.markdown("<style>" + (ROOT / "assets/styles.css").read_text(encoding="utf-8") + "</style>",
                unsafe_allow_html=True)


def sidebar():
    aliases = {"Meal Studio": "Add a meal", "Daily Log": "My day", "Nutrition Guide": "Food guide",
               "About MacroMate": "About"}
    current = aliases.get(st.session_state.get("workspace", "Home"), st.session_state.get("workspace", "Home"))
    st.session_state.workspace = current
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand"><img src="{asset_uri(ROOT / "assets/logo.svg")}" '
                    'alt="MacroMate logo"><div>MacroMate<small>KNOW YOUR PLATE</small></div></div>', unsafe_allow_html=True)
        for page, symbol in [("Home", "home"), ("Add a meal", "add_a_photo"), ("My day", "today")]:
            st.button(page, key="nav_" + page.lower().replace(" ", "_"), icon=f":material/{symbol}:",
                      type="primary" if current == page else "secondary", width="stretch", on_click=navigate, args=(page,))
        with st.expander("More"):
            st.button("Food guide", key="nav_guide", on_click=navigate, args=("Food guide",), width="stretch")
            st.button("About", key="nav_about", on_click=navigate, args=("About",), width="stretch")
        st.markdown('<div class="sidebar-note"><strong>Your log is temporary.</strong>'
                    '<p>Download your meals from My day to keep a copy.</p></div>', unsafe_allow_html=True)
    return current


def topbar(page):
    st.markdown(f'<div class="workspace-topbar"><span>MacroMate</span><span>{escape(page)}</span></div>', unsafe_allow_html=True)


def heading(eyebrow, title, description):
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div><h1 class="page-title">{escape(title)}</h1>'
                f'<p class="page-description">{escape(description)}</p>', unsafe_allow_html=True)


def metrics(values):
    with st.container(key="nutrition_metrics"):
        for column, (label, key, unit) in zip(st.columns(4), [
            ("Calories", "kcal", "kcal"), ("Protein", "protein_g", "g"),
            ("Carbs", "carbs_g", "g"), ("Fat", "fat_g", "g"),
        ]):
            column.metric(label, f"{values[key]:,.2f} {unit}")


def home(flow, labels, start_meal):
    with st.container(key="home_hero"):
        copy, visual = st.columns([1.2, 1], gap="large", vertical_alignment="center")
        with copy:
            st.markdown(f'<div class="hero-brand"><img src="{asset_uri(ROOT / "assets/logo.svg")}" '
                        'alt="MacroMate logo"><span>MacroMate</span></div>'
                        '<h1 class="hero-title">Know your meal.<br><em>In a few taps.</em></h1>'
                        '<p class="hero-description">Add a photo, check the dish and serving weight, '
                        'then save your meal. We’ll handle the maths.</p>', unsafe_allow_html=True)
            if flow.image_id is not None and not flow.added:
                st.button("Continue your meal", key="hero_start", type="primary", icon=":material/arrow_forward:",
                          on_click=navigate, args=("Add a meal",))
            else:
                st.button("Start a meal", key="hero_start", type="primary", icon=":material/arrow_forward:", on_click=start_meal)
            st.button("See my day", key="hero_day", on_click=navigate, args=("My day",))
        with visual:
            st.markdown(f'<div class="hero-visual"><img class="hero-meal" src="{asset_uri(ROOT / "assets/meal.jpg")}" '
                        'alt="A plate of chicken biryani"></div>', unsafe_allow_html=True)
    st.markdown('<div class="simple-how"><span><b>1</b> Add a photo</span><span><b>2</b> Check your portion</span>'
                '<span><b>3</b> Save your meal</span></div>', unsafe_allow_html=True)
    entries = flow.log.entries_for(date.today())
    if entries:
        st.success(f"You’ve saved {len(entries)} meal{'s' if len(entries) != 1 else ''} today.")
    with st.expander("Which foods are supported?"):
        st.write(", ".join(name(label) for label in labels) + ".")
        st.caption("One dish per photo. Nutrition is estimated from recipes and the grams you enter.")


def steps(current):
    items = []
    for index, label in enumerate(["Photo", "Dish & portion", "Review"]):
        state = "done" if index < current else "active" if index == current else ""
        items.append(f'<div class="workflow-step {state}"><span>{index+1}</span><b>{label}</b></div>')
    st.markdown('<div class="workflow-steps">' + "".join(items) + '</div>', unsafe_allow_html=True)


def rankings(suggestions):
    for item in suggestions:
        st.write(f"{name(item['dish'])} · ranking score {item['score']:.3g}")
