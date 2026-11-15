"""Shared presentation components. Static brand assets never cache user uploads."""

import base64
from datetime import date
from functools import lru_cache
from html import escape
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Home", "Meal Studio", "Nutrition Guide", "Daily Log", "About MacroMate"]
ICONS = {
    "scan": '<rect x="6" y="6" width="12" height="12" rx="6"/><path d="M8 2H4a2 2 0 0 0-2 2v4m14-6h4a2 2 0 0 1 2 2v4M2 16v4a2 2 0 0 0 2 2h4m8 0h4a2 2 0 0 0 2-2v-4"/>',
    "leaf": '<path d="M20 3C7 2 2 9 6 16s16 2 14-13Z"/><path d="M3 22 15 10"/>',
    "log": '<rect x="4" y="3" width="16" height="19" rx="3"/><path d="M8 1v4m8-4v4M8 10h8m-8 4h8m-8 4h4"/>',
    "plate": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/>',
    "arrow": '<path d="M5 12h14m-6-6 6 6-6 6"/>',
    "shield": '<path d="m12 2 8 3v6c0 5-4 9-8 11-4-2-8-6-8-11V5l8-3Z"/><path d="m8 12 3 3 5-6"/>',
}


def icon(name, size=24):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>')


def name(label):
    return label.replace("_", " ").title()


@lru_cache(maxsize=12)
def asset_uri(path):
    path = Path(path)
    mime = "image/svg+xml" if path.suffix == ".svg" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def dish_photo(label):
    return next(iter(sorted((ROOT / "data" / "images" / label).glob("*.jpg"))), None)


def navigate(page):
    st.session_state.workspace = page


def setup():
    st.markdown("<style>" + (ROOT / "assets" / "styles.css").read_text(encoding="utf-8") + "</style>",
                unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown(f'<div class="sidebar-brand"><img src="{asset_uri(ROOT / "assets/logo.svg")}" '
                    'alt="MacroMate logo"><div>MacroMate<small>KNOW YOUR PLATE</small></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="nav-label">YOUR WORKSPACE</div>', unsafe_allow_html=True)
        page = st.radio("Workspace", PAGES, key="workspace", label_visibility="collapsed")
        st.markdown(f'<div class="sidebar-note"><span>{icon("shield", 22)}</span>'
                    '<strong>A space just for this session.</strong><p>Your photos stay out of the log. '
                    'Export your meals before leaving to keep a copy.</p></div>'
                    '<div class="sidebar-bottom">MADE FOR FAMILIAR FOOD<br><b>Thoughtful choices, one meal at a time.</b></div>',
                    unsafe_allow_html=True)
    return page


def topbar(page):
    st.markdown(f'<div class="workspace-topbar"><span>WORKSPACE <b>/ {escape(page)}</b></span>'
                '<span class="session-pill"><i></i> Your private session</span></div>', unsafe_allow_html=True)


def heading(eyebrow, title, description):
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div><h1 class="page-title">{escape(title)}</h1>'
                f'<p class="page-description">{escape(description)}</p>', unsafe_allow_html=True)


def metrics(values):
    with st.container(key="nutrition_metrics"):
        for column, (label, key, unit) in zip(st.columns(4), [
            ("Energy", "kcal", "kcal"), ("Protein", "protein_g", "g"),
            ("Carbohydrates", "carbs_g", "g"), ("Fat", "fat_g", "g"),
        ]):
            column.metric(label, f"{values[key]:,.2f} {unit}")


def home(flow, labels):
    with st.container(key="home_hero"):
        copy, visual = st.columns([1.15, 1], gap="large", vertical_alignment="center")
        with copy:
            st.markdown(f'<div class="hero-brand"><img src="{asset_uri(ROOT / "assets/logo.svg")}" '
                        'alt="MacroMate logo"><span>MacroMate</span></div>'
                        '<div class="eyebrow">A FRESH WAY TO KNOW YOUR PLATE</div>'
                        '<h1 class="hero-title">Familiar food.<br><em>Fresh perspective.</em></h1>'
                        '<p class="hero-description">A photo is the starting point. Turn the meals you love '
                        'into a clearer picture of your nutrition, one serving at a time.</p>', unsafe_allow_html=True)
            st.button("Recognize a meal", key="hero_start", type="primary", icon=":material/arrow_forward:",
                      on_click=navigate, args=("Meal Studio",))
            st.markdown('<div class="hero-footnote">8 supported dishes <span>·</span> Your serving, your choice</div>',
                        unsafe_allow_html=True)
        with visual:
            photo = dish_photo("chicken_biryani")
            meal = f'<img class="hero-meal" src="{asset_uri(photo)}" alt="A plate of chicken biryani">' if photo else ""
            st.markdown(f'<div class="hero-visual"><div class="plate-orbit"></div><div class="plate-orbit inner"></div>'
                        f'{meal}<div class="floating-label label-top">{icon("scan", 20)}<div><b>Start with a photo</b>'
                        '<small>Keep the final say.</small></div></div>'
                        f'<div class="floating-label label-bottom"><span class="round-leaf">{icon("leaf", 22)}</span>'
                        '<div><b>Made for your everyday</b><small>Familiar dishes. Clear estimates.</small></div></div>'
                        '<span class="orbit-dot one"></span><span class="orbit-dot two"></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><div><span class="eyebrow">YOUR EVERYDAY TOOLKIT</span>'
                '<h2>A space for every step.</h2></div><span class="section-note">Choose where to begin.</span></div>',
                unsafe_allow_html=True)
    routes = [
        ("scan", "01", "Meal Studio", "Meet your meal.", "Upload a photo, review the suggestions, and confirm what is on your plate.", "Open Meal Studio"),
        ("leaf", "02", "Nutrition Guide", "Understand your serving.", "Explore the eight dish references, serving weights, and the sources behind the numbers.", "Explore nutrition"),
        ("log", "03", "Daily Log", "See your day clearly.", "Bring your meals together, check daily totals, and export a record to keep.", "View Daily Log"),
    ]
    for column, (symbol, number, page, title, description, action) in zip(st.columns(3, gap="medium"), routes):
        with column, st.container(border=True, key="route_" + number):
            st.markdown(f'<div class="route-top"><span class="icon-tile">{icon(symbol, 25)}</span><span>{number}</span></div>'
                        f'<h3 class="route-title">{title}</h3><p class="route-description">{description}</p>', unsafe_allow_html=True)
            st.button(action, key="route_button_" + number, on_click=navigate, args=(page,), width="stretch")
    left, right = st.columns([1.8, 1], gap="large")
    with left:
        st.markdown('<div class="section-heading compact"><div><span class="eyebrow">ROOTED IN FAMILIAR FLAVOURS</span>'
                    '<h2>Made for your kind of food.</h2></div></div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="dish-chip">{escape(name(label))}</span>' for label in labels)
        st.markdown(f'<div class="dish-chips">{chips}</div><p class="quiet-note">One supported dish per photo. '
                    'Serving weight is entered by you; recipes can change the estimate.</p>', unsafe_allow_html=True)
    with right, st.container(key="today_glance"):
        today = date.today()
        entries = flow.log.entries_for(today)
        st.markdown('<div class="eyebrow">TODAY, IN YOUR SESSION</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glance-number">{len(entries):02d}<span>meals logged</span></div>'
                    '<p class="quiet-note">Every meal you add helps build a clearer picture.</p>', unsafe_allow_html=True)
        st.button("Go to your daily log", on_click=navigate, args=("Daily Log",), width="stretch")


def steps(flow):
    current = 3 if flow.estimate else 2 if flow.confirmed_dish else 1 if flow.suggestions else 0
    items = []
    for index, label in enumerate(["Add a photo", "Confirm the dish", "Enter your serving", "Estimate & log"]):
        state = "done" if index < current else "active" if index == current else ""
        items.append(f'<div class="workflow-step {state}"><span>{index + 1:02d}</span><b>{label}</b></div>')
    st.markdown('<div class="workflow-steps">' + "".join(items) + '</div>', unsafe_allow_html=True)


def empty_suggestions():
    st.markdown(f'<div class="empty-suggestions"><span class="empty-icon">{icon("plate", 48)}</span>'
                '<h3>A little clarity starts here.</h3><p>Upload a meal photo and run recognition. '
                'Your three dish suggestions will appear in this space.</p>'
                '<div class="empty-detail"><b>You always have the final say.</b><br>'
                'Correct the dish, enter your own grams, or cancel an unsupported meal.</div></div>', unsafe_allow_html=True)


def rankings(suggestions):
    rows = []
    for index, item in enumerate(suggestions):
        label = escape(name(item["dish"]))
        rows.append(f'<div class="ranking-row {"first" if index == 0 else ""}"><span class="rank-number">0{index+1}</span>'
                    f'<div class="rank-dish">{label}{"<small>Top suggestion</small>" if index == 0 else ""}</div>'
                    f'<div class="rank-score">{item["score"]:.3g}<small>ranking score</small></div></div>')
    st.markdown('<div class="ranking-list">' + "".join(rows) + '</div>', unsafe_allow_html=True)
