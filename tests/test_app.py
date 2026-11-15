"""Real Streamlit widget interactions with only model/setup replaced by test doubles."""

import io
from pathlib import Path

from PIL import Image
from streamlit.testing.v1 import AppTest


def photo():
    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), "green").save(buffer, "PNG")
    buffer.name = "meal.png"
    buffer.seek(0)
    return buffer


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_ready_and_about_screens():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert button(app, "Find dish suggestions").disabled
    app.radio[0].set_value("About").run()
    assert not app.exception
    assert any("54 of 56" in item.value for item in app.markdown)


def test_full_ui_flow_reset_delete_and_export(monkeypatch):
    import streamlit as st
    from app_core import inference
    from scripts import prepare_app_assets

    class FakeModel:
        def predict(self, photo, prepared=False):
            return {"ranked_dishes": [{"dish": dish, "score": score} for dish, score in
                                      [("daal", .7), ("chicken biryani", .2), ("samosa", .1)]]}

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: photo())
    monkeypatch.setattr(prepare_app_assets, "prepare_assets", lambda: None)
    monkeypatch.setattr(inference, "get_recognizer", lambda *a, **k: FakeModel())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    button(app, "Find dish suggestions").click().run()
    assert not app.exception
    app.selectbox(key="dish").set_value("chicken_biryani").run()
    button(app, "Confirm dish").click().run()
    button(app, "Calculate nutrition").click().run()
    assert app.session_state.workflow.estimate["kcal"] == 160.335
    app.number_input(key="grams").set_value(200).run()
    assert app.session_state.workflow.estimate is None
    button(app, "Calculate nutrition").click().run()
    button(app, "Add to daily log").click().run()
    app.run()
    assert len(app.session_state.workflow.log.entries) == 1
    assert button(app, "Added to daily log").disabled
    app.radio[0].set_value("Daily log").run()
    assert not app.exception
    assert len(app.get("download_button")) == 2
    button(app, "Remove").click().run()
    assert len(app.session_state.workflow.log.entries) == 0
    app.radio[0].set_value("Log a meal").run()
    app.selectbox(key="dish").set_value("unsupported").run()
    assert app.session_state.workflow.estimate is None
    assert not app.exception
