"""MacroMate: photo recognition, confirmed serving estimates, and a private session log."""

import io
import logging
import time
from datetime import date

import streamlit as st
from PIL import Image, ImageOps

from app_core.inference import ImageError, check_image, predict_dishes
from app_core.nutrition import NutritionError, get_table
from app_core.workflow import MealWorkflow

st.set_page_config(page_title="MacroMate | Your meal, understood", page_icon="🌿", layout="wide")
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
    columns = st.columns(4)
    for column, (label, key, unit) in zip(columns, [
        ("Energy", "kcal", "kcal"), ("Protein", "protein_g", "g"),
        ("Carbohydrates", "carbs_g", "g"), ("Fat", "fat_g", "g"),
    ]):
        column.metric(label, f"{values[key]:,.2f} {unit}")


st.markdown("""
<style>
  .stApp {background:#f6f8f5;}
  .block-container {max-width:1240px; padding-top:2.5rem; padding-bottom:3rem;}
  h1,h2,h3 {color:#153c2b; letter-spacing:-.035em;}
  h1 {font-size:2.7rem !important; line-height:1.12 !important;}
  [data-testid="stSidebar"] {background:#edf3ed;}
  [data-testid="stVerticalBlockBorderWrapper"] {background:white; border-radius:18px;}
  [data-testid="stMetric"] {background:#edf5ee; padding:16px; border-radius:12px;}
  [data-testid="stMetricLabel"] {color:#466252;}
  [data-testid="stMetricValue"] {font-size:1.7rem; color:#174e35;}
  .eyebrow {color:#28714c; font-size:.78rem; font-weight:700; letter-spacing:.16em;}
  .intro {color:#5b6d62; font-size:1.05rem; max-width:640px; margin-bottom:1.4rem;}
  .brand {font-size:1.6rem; font-weight:750; color:#16452e; margin-bottom:.2rem;}
  @media(max-width:640px) {h1 {font-size:2rem !important;} .block-container {padding-top:1.5rem;}}
</style>
""", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown('<div class="brand">🌿 MacroMate</div>', unsafe_allow_html=True)
    st.caption("A little more clarity in every meal.")
    st.divider()
    page = st.radio("Workspace", ["Log a meal", "Daily log", "About"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Made for eight dishes**")
    st.caption(" · ".join(display_name(label) for label in labels))
    st.info("Your log is temporary and private to this session. Export it before leaving.")

if page == "Log a meal":
    st.markdown('<div class="eyebrow">PHOTO → CONFIRM → ESTIMATE → LOG</div>', unsafe_allow_html=True)
    st.title("Your meal, understood.")
    st.markdown('<p class="intro">Start with a photo. Confirm the dish, enter your serving weight, '
                'and keep a simple record of your day.</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.1], gap="large")
    with left, st.container(border=True):
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
            except FileNotFoundError:
                logger.exception("Model assets missing")
                st.error("The recognizer is not ready. The maintainer needs to provision the model assets.")
            except Exception:
                logger.exception("Prediction failed (photo content is not logged)")
                st.error("We could not recognize this photo. Try another supported meal photo or try again later.")
    with right, st.container(border=True):
        st.subheader("2. Make it your meal")
        if not flow.suggestions:
            st.info("Your dish suggestions will appear here after you upload a photo and run recognition.")
            st.markdown("**You stay in control**")
            st.write("Confirm the dish yourself and enter grams from a scale or your own estimate.")
            st.caption("The photo recognizer does not measure portion weight or detect unfamiliar foods.")
        else:
            st.caption("TOP THREE SUGGESTIONS · UNCALIBRATED RANKING SCORES")
            for rank, suggestion in enumerate(flow.suggestions, 1):
                a, b = st.columns([3, 1])
                a.markdown(f"**{rank}. {display_name(suggestion['dish'])}**")
                b.write(f"{suggestion['score']:.3f}")
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
                    except NutritionError as error:
                        st.error(str(error))
    if flow.estimate:
        with st.container(border=True):
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
                st.success("Meal added. Open Daily log to see totals, remove an entry, or export your record.")

elif page == "Daily log":
    st.markdown('<div class="eyebrow">YOUR SESSION, AT A GLANCE</div>', unsafe_allow_html=True)
    st.title("A clearer picture of your day.")
    st.caption("Temporary session log. Photos are not saved to disk. Export your entries before closing this session.")
    day = st.date_input("View meals for", value=date.today(), key="log_date")
    entries = flow.log.entries_for(day)
    metrics(flow.log.totals_for(day))
    st.subheader(f"Meals · {len(entries)}")
    if not entries:
        st.info("No meals logged for this date. Start in Log a meal to add your first serving.")
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

else:
    st.markdown('<div class="eyebrow">A SMALL MODEL, WITH CLEAR LIMITS</div>', unsafe_allow_html=True)
    st.title("Know what goes into your estimate.")
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
