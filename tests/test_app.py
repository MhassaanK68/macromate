"""Guided UI journeys; model calls are replaced only inside these tests."""

import io
from datetime import date
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


def new_app():
    return AppTest.from_file(str(APP), default_timeout=15).run()


@pytest.fixture
def fake_upload(monkeypatch):
    import streamlit as st
    from app_core import inference
    from scripts import prepare_app_assets

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "green").save(buffer, "PNG")
    upload = {"content": buffer.getvalue(), "calls": 0, "fail": False}

    def uploaded(*args, **kwargs):
        if upload["content"] is None:
            return None
        file = io.BytesIO(upload["content"])
        file.name = "meal.png"
        return file

    class FakeModel:
        def predict(self, photo, prepared=False):
            upload["calls"] += 1
            if upload["fail"]:
                raise RuntimeError("Test-only model failure")
            return {"ranked_dishes": [{"dish": dish, "score": score} for dish, score in
                                      [("daal", .7), ("chicken biryani", .2), ("samosa", .1)]]}

    st.cache_resource.clear()
    monkeypatch.setattr(st, "file_uploader", uploaded)
    monkeypatch.setattr(prepare_app_assets, "prepare_assets", lambda: None)
    monkeypatch.setattr(inference, "get_recognizer", lambda *a, **k: FakeModel())
    return upload


def test_home_has_one_start_action_and_simple_navigation():
    app = new_app()
    assert not app.exception
    assert app.session_state.workspace == "Home"
    button(app, "Start a meal").click().run()
    assert app.session_state.workspace == "Add a meal"
    assert len(app.get("file_uploader")) == 1
    assert not any(item.label in ("Find dish suggestions", "Confirm dish", "Calculate nutrition") for item in app.button)
    button(app, "About").click().run()
    assert any("54 of 56" in item.value for item in app.markdown)


def test_automatic_recognition_combined_confirmation_and_single_save(fake_upload):
    app = new_app()
    button(app, "Start a meal").click().run()
    assert fake_upload["calls"] == 1
    assert len(app.metric) == 0  # No nutrition before explicit confirmation.
    app.selectbox(key="_meal_dish").set_value("chicken_biryani").run()
    button(app, "Confirm & see nutrition").click().run()
    assert app.metric[0].value == "160.34 kcal"
    assert app.session_state.workflow.confirmed_dish == "chicken_biryani"
    button(app, "Save meal").click().run()
    app.run()
    assert len(app.session_state.workflow.log.entries) == 1
    assert not any(item.label == "Save meal" for item in app.button)
    assert button(app, "See my day")
    assert fake_upload["calls"] == 1
    assert not app.exception


def test_hosted_uploader_uses_request_safe_limit(monkeypatch):
    monkeypatch.setenv("MACROMATE_UPLOAD_LIMIT_MB", "4")
    app = new_app()
    button(app, "Start a meal").click().run()
    assert app.get("file_uploader")[0].proto.max_upload_size_mb == 4
    assert any("up to 4 MB" in item.value for item in app.caption)
    assert not app.exception


def test_edit_and_navigation_preserve_pending_meal(fake_upload):
    app = new_app()
    button(app, "Start a meal").click().run()
    app.selectbox(key="_meal_dish").set_value("chicken_biryani").run()
    app.number_input(key="_meal_grams").set_value(200).run()
    button(app, "Confirm & see nutrition").click().run()
    button(app, "Home").click().run()
    button(app, "Continue your meal").click().run()
    assert app.session_state.workflow.estimate["serving_grams"] == 200
    button(app, "Edit dish or portion").click().run()
    assert app.session_state.workflow.estimate is None
    assert app.number_input(key="_meal_grams").value == 200
    assert app.selectbox(key="_meal_dish").value == "chicken_biryani"
    app.number_input(key="_meal_grams").set_value(150).run()
    button(app, "Food guide").click().run()
    button(app, "Back to my meal").click().run()
    button(app, "Confirm & see nutrition").click().run()
    assert app.session_state.workflow.estimate["kcal"] == pytest.approx(160.335)
    assert fake_upload["calls"] == 1
    assert not app.exception


def test_remove_undo_download_and_new_meal(fake_upload):
    app = new_app()
    button(app, "Start a meal").click().run()
    button(app, "Confirm & see nutrition").click().run()
    app.date_input(key="_meal_date").set_value(None).run()
    assert button(app, "Save meal").disabled
    app.date_input(key="_meal_date").set_value(date(2026, 1, 2)).run()
    button(app, "Save meal").click().run()
    button(app, "See my day").click().run()
    assert app.date_input(key="log_date").value == date(2026, 1, 2)
    assert len(app.metric) == 4
    assert not app.warning
    assert len(app.get("download_button")) == 2
    button(app, "Remove").click().run()
    assert app.session_state.workflow.log.entries == []
    button(app, "Undo").click().run()
    assert len(app.session_state.workflow.log.entries) == 1
    fake_upload["content"] = None
    button(app, "Add another meal").click().run()
    assert app.session_state.workflow.suggestions == []
    assert len(app.session_state.workflow.log.entries) == 1
    assert not app.exception


def test_failed_recognition_waits_for_explicit_retry(fake_upload):
    fake_upload["fail"] = True
    app = new_app()
    button(app, "Start a meal").click().run()
    app.run()
    assert fake_upload["calls"] == 1
    assert app.error
    fake_upload["fail"] = False
    button(app, "Try again").click().run()
    assert fake_upload["calls"] == 2
    assert app.selectbox(key="_meal_dish")
    assert not app.exception


def test_unsupported_and_invalid_servings_cannot_be_saved(fake_upload):
    app = new_app()
    button(app, "Start a meal").click().run()
    app.number_input(key="_meal_grams").set_value(0).run()
    assert button(app, "Confirm & see nutrition").disabled
    app.selectbox(key="_meal_dish").set_value("unsupported").run()
    button(app, "Home").click().run()
    button(app, "Continue your meal").click().run()
    assert app.selectbox(key="_meal_dish").value == "unsupported"
    assert not any(item.label in ("Confirm & see nutrition", "Save meal") for item in app.button)
    assert app.session_state.workflow.estimate is None
    fake_upload["content"] = None
    button(app, "Change photo").click().run()
    assert app.session_state.workflow.image_id is None
    assert not app.exception


def test_corrupt_photo_rejected_before_model(fake_upload):
    fake_upload["content"] = b"not a readable image"
    app = new_app()
    button(app, "Start a meal").click().run()
    assert app.error and fake_upload["calls"] == 0
    assert app.session_state.workflow.estimate is None
    assert not app.exception


def test_food_guide_does_not_log_and_empty_day_stays_simple():
    app = new_app()
    button(app, "Food guide").click().run()
    app.selectbox(key="guide_dish").set_value("chicken_biryani").run()
    app.number_input(key="guide_grams").set_value(150).run()
    assert app.metric[0].value == "160.34 kcal"
    assert app.session_state.workflow.log.entries == []
    button(app, "My day").click().run()
    assert len(app.metric) == 0
    assert len(app.get("download_button")) == 0
    assert button(app, "Start a meal")
    app.date_input(key="log_date").set_value(None).run()
    assert app.info and len(app.metric) == 0
    assert not app.exception
