"""MacroMate: a guided photo, portion, and save flow with a session-local log."""

import io
import logging
import os
from datetime import date

import streamlit as st
from PIL import Image, ImageOps

from app_core import ui
from app_core.inference import ImageError, check_image, predict_dishes
from app_core.nutrition import NutritionError, calculate_nutrition, get_table
from app_core.workflow import MealWorkflow

st.set_page_config(page_title="MacroMate | Simple meal logging",
                   page_icon=str(ui.ROOT / "assets/logo.svg"), layout="wide")
logger = logging.getLogger("macromate.app")


@st.cache_resource(show_spinner=False)
def prepare_recognizer():
    from scripts.prepare_app_assets import prepare_assets
    from app_core.inference import get_recognizer

    prepare_assets()
    return get_recognizer()


ui.setup()
if "workflow" not in st.session_state:
    st.session_state.workflow = MealWorkflow()
flow = st.session_state.workflow


def start_meal():
    """Reset the pending meal, preserving saved meals and the latest undo."""
    flow.set_image(None)
    for key in ("meal_photo", "meal_draft", "recognition_error", "prediction_attempt",
                "_meal_dish", "_meal_grams", "_meal_date"):
        st.session_state.pop(key, None)
    st.session_state.upload_version = st.session_state.get("upload_version", 0) + 1
    ui.navigate("Add a meal")


def edit_meal():
    flow.cancel()


def upload_changed(key):
    """Respect the uploader's remove action without losing drafts on navigation."""
    if st.session_state.get(key) is None:
        flow.set_image(None)
        for name in ("meal_photo", "meal_draft", "recognition_error", "prediction_attempt"):
            st.session_state.pop(name, None)


def recognize(content):
    st.session_state.prediction_attempt = flow.image_id
    st.session_state.pop("recognition_error", None)
    try:
        with st.spinner("Looking at your meal… The first photo may take a little longer."):
            prepare_recognizer()
            suggestions = predict_dishes(content)
            if not suggestions:
                raise ValueError("No dish suggestions returned")
            flow.set_predictions(suggestions)
            st.session_state.meal_draft = {"dish": suggestions[0]["label"], "grams": 150.0,
                                           "date": date.today()}
        st.rerun()
    except Exception:
        logger.exception("Recognition failed; photo content is not logged")
        st.session_state.recognition_error = (
            "We couldn’t check this photo just now. Try again, or choose a different photo."
        )


def preview_photo(width=220):
    content = st.session_state.get("meal_photo")
    if content:
        with Image.open(io.BytesIO(content)) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((700, 500))
            st.image(preview, width=width)


def add_meal_screen(labels):
    if flow.added:
        result = flow.estimate
        ui.heading("ALL DONE", "Meal saved.", "You’re all set. Your daily totals have been updated.")
        with st.container(border=True, key="saved_panel"):
            st.markdown(f'<div class="saved-mark">{ui.icon("check", 34)}</div>', unsafe_allow_html=True)
            st.subheader(ui.name(result["dish"]))
            st.write(f"{result['serving_grams']:g} g · {result['kcal']:.2f} kcal")
            saved = next(entry for entry in flow.log.entries if entry["entry_id"] == flow.entry_id) if any(
                entry["entry_id"] == flow.entry_id for entry in flow.log.entries) else None
            if saved:
                st.caption("Saved for " + saved["meal_date"].strftime("%d %B %Y"))
            st.button("See my day", type="primary", width="stretch", on_click=ui.navigate, args=("My day",))
            st.button("Add another meal", width="stretch", on_click=start_meal)
        return

    stage = 2 if flow.estimate else 1 if flow.suggestions else 0
    titles = ["Let’s add your meal.", "Check your dish and portion.", "Ready to save?"]
    descriptions = ["Choose a clear photo of one dish. We’ll look for it automatically.",
                    "Check the suggested dish, then tell us how much you’re having.",
                    "Here’s the estimate for your serving. Save it when it looks right."]
    ui.heading("ADD A MEAL", titles[stage], descriptions[stage])
    ui.steps(stage)
    with st.container(border=True, key="meal_flow"):
        if stage == 0:
            upload_key = f"photo_{st.session_state.get('upload_version', 0)}"
            limit = min(10, int(os.environ.get("MACROMATE_UPLOAD_LIMIT_MB", "10")),
                        st.get_option("server.maxUploadSize"))
            upload = st.file_uploader("Add a meal photo", type=["jpg", "jpeg", "png"],
                                      key=upload_key, on_change=upload_changed, args=(upload_key,),
                                      max_upload_size=limit)
            st.caption(f"JPEG or PNG · up to {limit:g} MB. One of our eight supported dishes per photo.")
            content = upload.getvalue() if upload is not None else st.session_state.get("meal_photo")
            if content is not None:
                try:
                    check_image(content)
                    if flow.set_image(content):
                        for key in ("recognition_error", "prediction_attempt", "meal_draft"):
                            st.session_state.pop(key, None)
                    st.session_state.meal_photo = content
                    if st.session_state.get("prediction_attempt") != flow.image_id:
                        recognize(content)
                    if st.session_state.get("recognition_error"):
                        preview_photo()
                        st.error(st.session_state.recognition_error)
                        if st.button("Try again", type="primary"):
                            recognize(content)
                        st.button("Choose another photo", on_click=start_meal)
                except ImageError as error:
                    flow.set_image(None)
                    st.session_state.pop("meal_photo", None)
                    st.error(str(error) + ". Please choose another photo.")
            with st.expander("Which foods can I use?"):
                st.write(", ".join(ui.name(label) for label in labels) + ".")
                st.caption("Mixed plates and other foods are not supported yet.")

        elif stage == 1:
            photo, fields = st.columns([1, 2], gap="large")
            draft = st.session_state.meal_draft
            with photo:
                preview_photo(width="stretch")
                st.button("Change photo", on_click=start_meal, width="stretch")
            with fields:
                st.session_state.setdefault("_meal_dish", draft["dish"])
                st.session_state.setdefault("_meal_grams", draft["grams"])
                dish = st.selectbox("What is this dish?", labels + ["unsupported"], key="_meal_dish",
                                    format_func=lambda value: "My dish isn’t listed" if value == "unsupported" else ui.name(value))
                draft["dish"] = dish
                st.caption("Not quite right? Choose the correct dish from the list.")
                if dish == "unsupported":
                    flow.cancel()
                    st.info("We don’t have nutrition for this food yet. Choose a photo of a supported dish to continue.")
                else:
                    grams = st.number_input("How much are you having? (grams)", min_value=0.0,
                                            max_value=5000.0, step=25.0, key="_meal_grams",
                                            help="Enter the edible portion’s weight. A kitchen scale gives the best estimate.")
                    draft.update(dish=dish, grams=grams)
                    st.caption("Enter grams from a scale or your own estimate. The photo cannot measure weight.")
                    if grams <= 0:
                        st.warning("Enter a serving weight greater than zero.")
                    if st.button("Confirm & see nutrition", type="primary", width="stretch", disabled=grams <= 0):
                        try:
                            flow.confirm(dish)
                            flow.calculate(grams)
                            st.rerun()
                        except NutritionError as error:
                            st.error(str(error))
                with st.expander("Other dish suggestions"):
                    ui.rankings(flow.suggestions)
                    st.caption("These scores rank eight dishes. They are not verified confidence, and unfamiliar foods may still receive a suggestion.")

        else:
            result = flow.estimate
            st.subheader(f"{ui.name(result['dish'])} · {result['serving_grams']:g} g")
            ui.metrics(result)
            st.caption("An estimate for this recipe. Ingredients and cooking methods can change the values.")
            draft = st.session_state.meal_draft
            with st.expander("Change the meal date"):
                meal_date = st.date_input("Meal date", value=draft["date"], key="_meal_date")
                draft["date"] = meal_date
            with st.expander("Recipe details and source"):
                st.write(result["recipe_assumption"])
                st.link_button("Read the nutrition source", result["source"])
                st.caption("Reference quality: " + result["quality_flag"].replace("_", " "))
            if meal_date is None:
                st.warning("Choose a meal date before saving.")
            else:
                st.caption("Saving for " + meal_date.strftime("%d %B %Y"))
            if st.button("Save meal", type="primary", width="stretch", disabled=meal_date is None):
                flow.add(meal_date)
                st.session_state.log_date = meal_date
                st.rerun()
            st.button("Edit dish or portion", width="stretch", on_click=edit_meal)


try:
    table = get_table()
except (NutritionError, OSError, KeyError):
    logger.exception("Nutrition reference unavailable")
    st.error("Nutrition is unavailable just now. Please try again later.")
    st.stop()
labels = list(table)
page = ui.sidebar()
ui.topbar(page)

if page == "Home":
    ui.home(flow, labels, start_meal)
elif page == "Add a meal":
    add_meal_screen(labels)
elif page == "My day":
    ui.heading("MY DAY", "Your meals, together.", "See what you’ve saved and take a copy with you.")
    st.session_state.setdefault("log_date", date.today())
    day = st.date_input("Show meals for", value=None, key="log_date")
    if day is None:
        st.info("Choose a date to see your meals.")
        st.stop()
    entries = flow.log.entries_for(day)
    if flow.removed_entry is not None:
        notice, undo = st.columns([3, 1])
        notice.success("Meal removed.")
        undo.button("Undo", on_click=flow.undo_remove, width="stretch")
    if not entries:
        st.info("No meals saved for this day yet. Add your first meal to get started.")
        if flow.image_id is not None and not flow.added:
            st.button("Back to my meal", type="primary", on_click=ui.navigate, args=("Add a meal",))
        else:
            st.button("Start a meal", type="primary", on_click=start_meal)
    else:
        ui.metrics(flow.log.totals_for(day))
        st.button("Add another meal", type="primary", on_click=start_meal)
        st.subheader(f"{len(entries)} meal{'s' if len(entries) != 1 else ''}")
        for entry in entries:
            with st.container(border=True):
                detail, action = st.columns([5, 1])
                detail.markdown(f"**{ui.name(entry['dish'])}** · {entry['serving_grams']:g} g")
                detail.caption(f"{entry['kcal']:.2f} kcal · Protein {entry['protein_g']:.2f} g · "
                               f"Carbs {entry['carbs_g']:.2f} g · Fat {entry['fat_g']:.2f} g")
                action.button("Remove", key="remove_" + entry["entry_id"], on_click=flow.remove, args=(entry["entry_id"],))
        st.download_button("Download this day", flow.log.to_csv(day, all_dates=False),
                           file_name=f"macromate-{day.isoformat()}.csv", mime="text/csv")
        st.caption("Your log is temporary. Download a copy to keep it.")
    if flow.log.entries:
        with st.expander("Download all saved meals"):
            st.download_button("Download all meals", flow.log.to_csv(), file_name="macromate-meals.csv", mime="text/csv")

elif page == "Food guide":
    ui.heading("FOOD GUIDE", "Get to know your food.", "Pick a dish and a serving size to explore its nutrition.")
    with st.container(border=True, key="reference_panel"):
        left, right = st.columns(2)
        dish = left.selectbox("Choose a dish", labels, format_func=ui.name, key="guide_dish")
        grams = right.number_input("Serving size (grams)", min_value=1.0, max_value=5000.0,
                                    value=100.0, step=25.0, key="guide_grams")
        result = calculate_nutrition(dish, grams)
        ui.metrics(result)
        st.caption("Recipe estimate only. Exploring this page does not save a meal.")
        with st.expander("Recipe details and source"):
            st.write(result["recipe_assumption"])
            st.link_button("Read the nutrition source", result["source"])
        if flow.image_id is not None and not flow.added:
            st.button("Back to my meal", type="primary", on_click=ui.navigate, args=("Add a meal",))
        else:
            st.button("Start a meal", type="primary", on_click=start_meal)
else:
    ui.heading("ABOUT", "A little help with everyday meals.", "MacroMate suggests the dish. You make the final choice.")
    st.write("Upload one supported dish, check the name and serving weight, then save its estimated nutrition to your day.")
    st.subheader("What to keep in mind")
    st.write("We recognize eight dishes, not mixed plates or every kind of food. Serving weight is entered by you. "
             "Nutrition comes from recipe references and can vary with ingredients and cooking methods.")
    st.write("Your photos and unfinished meal stay in this session’s memory. Photos are not saved to disk or exported. "
             "The meal log can reset when the session ends; download it to keep a copy.")
    with st.expander("Model results and data sources"):
        st.write("The recorded experiment classified 54 of 56 held-out photos correctly (96.43%). "
                 "There were seven test photos per dish. This does not establish accuracy on your uploads, "
                 "and ranking scores are uncalibrated.")
        st.caption("Illustrative meal photos come from the project dataset. Source attribution is retained in data/metadata.csv and data/sources.csv.")
        st.link_button("View project and methodology", "https://github.com/MhassaanK68/macromate")
