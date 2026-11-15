"""MacroMate: photo recognition, confirmed serving estimates, and a private session log."""

import io
import logging
import time
from datetime import date

import streamlit as st
from PIL import Image, ImageOps

from app_core.inference import ImageError, check_image, predict_dishes
from app_core.nutrition import NutritionError, calculate_nutrition, get_table
from app_core.workflow import MealWorkflow
from app_core import ui

st.set_page_config(page_title="MacroMate | Know your plate", page_icon=str(ui.ROOT / "assets/logo.svg"), layout="wide")
logger = logging.getLogger("macromate.app")


@st.cache_resource(show_spinner=False)
def prepare_recognizer():
    # Community Cloud starts with a fresh filesystem. Download public assets once
    # per process, verify them, then let the adapter share its locked recognizer.
    from scripts.prepare_app_assets import prepare_assets
    from app_core.inference import get_recognizer

    prepare_assets()
    return get_recognizer()


def display_name(label):
    return label.replace("_", " ").title()


def metrics(values):
    ui.metrics(values)


ui.setup()

if "workflow" not in st.session_state:
    st.session_state.workflow = MealWorkflow()
flow = st.session_state.workflow

try:
    table = get_table()
except (NutritionError, OSError, KeyError):
    logger.exception("Nutrition reference unavailable")
    st.error("The nutrition reference is unavailable. Please contact the app maintainer.")
    st.stop()
labels = list(table)

page = ui.sidebar()
ui.topbar(page)

if page == "Home":
    ui.home(flow, labels)

elif page == "Meal Studio":
    ui.heading("MEAL STUDIO", "Your meal, in focus.",
               "A photo, your confirmation, and a serving weight. Make the estimate your own.")
    ui.steps(flow)
    left, right = st.columns([1, 1.1], gap="large")
    with left, st.container(border=True, key="upload_panel"):
        st.markdown('<div class="studio-label">START WITH WHAT IS ON YOUR PLATE</div>', unsafe_allow_html=True)
        st.subheader("1. Add your meal photo")
        st.caption("One supported dish per photo · JPEG or PNG · Up to 10 MB / 25 MP")
        upload = st.file_uploader("Choose a meal photo", type=["jpg", "jpeg", "png"], key="photo")
        content = upload.getvalue() if upload is not None else None
        if flow.set_image(content):
            for key in ("dish", "grams", "prediction_seconds"):
                st.session_state.pop(key, None)
        valid = False
        if content:
            try:
                check_image(content)
                with Image.open(io.BytesIO(content)) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((900, 650))
                    st.image(preview, caption=upload.name, width="stretch")
                valid = True
            except ImageError as error:
                flow.set_image(None)
                st.error(str(error))
        if st.button("Find dish suggestions", type="primary", disabled=not valid, width="stretch"):
            # Clear old predictions even if this inference attempt fails.
            flow.set_predictions([])
            try:
                with st.spinner("Looking at your meal… The first prediction can take longer."):
                    prepare_recognizer()
                    begin = time.perf_counter()
                    suggestions = predict_dishes(content)
                    st.session_state.prediction_seconds = time.perf_counter() - begin
                    flow.set_predictions(suggestions)
                    st.session_state.dish = suggestions[0]["label"]
                    st.rerun()
            except FileNotFoundError:
                logger.exception("Model assets missing")
                st.error("The recognizer is not ready. The maintainer needs to provision the model assets.")
            except Exception:
                logger.exception("Prediction failed (photo content is not logged)")
                st.error("We could not recognize this photo. Try another supported meal photo or try again later.")
    with right, st.container(border=True, key="confirm_panel"):
        st.markdown('<div class="studio-label">RECOGNITION, WITH YOU IN CONTROL</div>', unsafe_allow_html=True)
        st.subheader("2. Make it your meal")
        if not flow.suggestions:
            ui.empty_suggestions()
        else:
            st.caption("THREE SUGGESTIONS · YOU CHOOSE THE DISH")
            ui.rankings(flow.suggestions)
            st.caption(f"Recognition: {st.session_state.get('prediction_seconds', 0):.2f} s. "
                       "Scores rank these eight dishes; they are not verified confidence.")
            st.divider()
            dish = st.selectbox("Which dish is this?", labels + ["unsupported"],
                                format_func=lambda x: "Not supported / cancel" if x == "unsupported" else display_name(x),
                                key="dish")
            if dish == "unsupported":
                flow.cancel()
                st.warning("This meal will not be estimated or logged. Upload a photo of a supported dish to continue.")
            else:
                grams = st.number_input("Serving weight (grams)", min_value=0.0, max_value=5000.0,
                                        value=150.0, step=0.5, key="grams",
                                        help="Enter the edible serving weight yourself; 0 is not a valid serving.")
                flow.sync_inputs(dish, grams)
                if st.button("Confirm dish", disabled=flow.confirmed_dish == dish, width="stretch"):
                    flow.confirm(dish)
                    st.rerun()
                if flow.confirmed_dish:
                    st.success(f"Confirmed: {display_name(flow.confirmed_dish)}")
                if st.button("Calculate nutrition", type="primary", disabled=not flow.confirmed_dish or grams <= 0,
                             width="stretch"):
                    try:
                        flow.calculate(grams)
                        st.rerun()
                    except NutritionError as error:
                        st.error(str(error))
    if flow.estimate:
        with st.container(border=True, key="result_panel"):
            result = flow.estimate
            st.subheader("3. Your serving estimate")
            st.write(f"**{display_name(result['dish'])} · {result['serving_grams']:g} g**")
            metrics(result)
            st.caption("Estimated from a per-100-g reference. Recipes and cooking methods can change these values.")
            with st.expander("Recipe assumptions and nutrition source"):
                st.write(result["recipe_assumption"])
                st.write("Reference quality: " + result["quality_flag"].replace("_", " "))
                st.link_button(f"Open source {result['source_id']}", result["source"])
            meal_date = st.date_input("Meal date", value=date.today(), key="meal_date", disabled=flow.added)
            if st.button("Added to daily log" if flow.added else "Add to daily log", type="primary", disabled=flow.added):
                flow.add(meal_date)
                st.rerun()
            if flow.added:
                st.success("Meal added. Open Daily Log to see totals, remove an entry, or export your record.")
                st.button("View your daily log", on_click=ui.navigate, args=("Daily Log",), icon=":material/arrow_forward:")

elif page == "Daily Log":
    ui.heading("DAILY LOG", "The bigger picture, one meal at a time.",
               "Your meals, daily totals, and a record you can take with you.")
    st.caption("Temporary session log. Photos are not saved to disk. Export your entries before closing this session.")
    day = st.date_input("View meals for", value=date.today(), key="log_date")
    entries = flow.log.entries_for(day)
    metrics(flow.log.totals_for(day))
    st.subheader(f"Meals · {len(entries)}")
    if not entries:
        st.markdown(f'<div class="empty-suggestions"><span class="empty-icon">{ui.icon("log", 44)}</span>'
                    '<h3>A fresh page for your day.</h3><p>No meals are logged for this date yet. '
                    'Visit Meal Studio to add your first serving.</p></div>', unsafe_allow_html=True)
        st.button("Add your first meal", on_click=ui.navigate, args=("Meal Studio",), type="primary")
    for entry in entries:
        with st.container(border=True):
            detail, action = st.columns([5, 1])
            detail.markdown(f"**{display_name(entry['dish'])}** · {entry['serving_grams']:g} g")
            detail.caption(f"{entry['kcal']:.2f} kcal · Protein {entry['protein_g']:.2f} g · "
                           f"Carbs {entry['carbs_g']:.2f} g · Fat {entry['fat_g']:.2f} g · {entry['source_id']}")
            if action.button("Remove", key="remove_" + entry["entry_id"]):
                flow.log.remove(entry["entry_id"])
                st.rerun()
    a, b = st.columns(2)
    a.download_button("Export selected date", flow.log.to_csv(day, all_dates=False),
                      file_name=f"macromate-{day.isoformat()}.csv", mime="text/csv", disabled=not entries)
    b.download_button("Export all dates", flow.log.to_csv(), file_name="macromate-meals.csv",
                      mime="text/csv", disabled=not flow.log.entries)

elif page == "Nutrition Guide":
    ui.heading("NUTRITION GUIDE", "A closer look at your serving.",
               "Explore the dish references and recipe assumptions behind your nutrition estimates.")
    controls, reference = st.columns([1, 1.65], gap="large")
    with controls, st.container(border=True, key="reference_panel"):
        st.subheader("Reference calculator")
        guide_dish = st.selectbox("Explore a dish", labels, format_func=display_name, key="guide_dish")
        guide_grams = st.number_input("Reference serving (grams)", min_value=1.0, max_value=5000.0,
                                      value=100.0, step=25.0, key="guide_grams")
        photo = ui.dish_photo(guide_dish)
        if photo:
            st.markdown(f'<div class="reference-photo-frame"><img class="reference-photo" src="{ui.asset_uri(photo)}" '
                        f'alt="Illustrative {display_name(guide_dish)} photo"></div>', unsafe_allow_html=True)
        st.caption("Illustrative dataset photo. Recipes and ingredients can vary.")
    with reference:
        result = calculate_nutrition(guide_dish, guide_grams)
        st.markdown(f'<span class="reference-badge">REFERENCE ESTIMATE · {guide_grams:g} g</span>'
                    f'<div class="reference-heading">{display_name(guide_dish)}</div>', unsafe_allow_html=True)
        metrics(result)
        st.caption("Per-100-g reference × your selected grams ÷ 100. This explorer does not add a meal to your log.")
        st.divider()
        st.subheader("Know the recipe behind the numbers")
        st.write(result["recipe_assumption"])
        st.caption("Reference quality: " + result["quality_flag"].replace("_", " "))
        st.link_button(f"Read nutrition source · {result['source_id']}", result["source"])
        st.info("A dish name cannot capture every recipe. Oil, ingredients, cooking method, and serving weight all affect the estimate.")
        st.button("Recognize and log a meal", on_click=ui.navigate, args=("Meal Studio",), type="primary")

else:
    ui.heading("ABOUT MACROMATE", "Familiar meals. Thoughtful choices.",
               "A practical food-recognition prototype, with the assumptions kept in view.")
    st.markdown(f'<div class="about-brand"><img src="{ui.asset_uri(ui.ROOT / "assets/logo.svg")}" '
                'alt="MacroMate logo"><div><strong>MacroMate</strong><small>KNOW YOUR PLATE</small></div></div>',
                unsafe_allow_html=True)
    st.write("MacroMate is an academic prototype built around a partially fine-tuned CLIP ViT-B/32 recognizer.")
    st.subheader("Supported dishes")
    st.write(", ".join(display_name(label) for label in labels) + ".")
    st.subheader("How it works")
    st.write("Photo → ranked dish suggestions → your confirmation → manually entered grams → reference nutrition → session log.")
    st.subheader("Recognition and nutrition limits")
    st.write("The recorded experiment classified 54 of 56 held-out photos correctly (96.43%). "
             "There were only seven test photos per dish; this does not establish accuracy on your uploads. "
             "Mixed plates and unfamiliar foods are unsupported. Ranking scores are uncalibrated.")
    st.write("Nutrition comes from recipe-specific reference values. Nihari uses a chicken proxy, "
             "samosa uses an oven-heated vegetable proxy, and biryani uses a lower-oil recipe. "
             "These estimates are not personalized diet advice.")
    st.subheader("Your data")
    st.write("Uploads and results are processed in memory within this session. Photos are not written to disk "
             "or added to the log. Logs can disappear after refresh, disconnection, or server restart. "
             "Export CSV to keep your record. Independent sessions have separate logs.")
    st.link_button("View project and methodology", "https://github.com/MhassaanK68/macromate")
    st.caption("Illustrative meal photos come from the project dataset. Source records and uploader-stated CC BY 4.0 attribution are retained in data/metadata.csv and data/sources.csv.")
