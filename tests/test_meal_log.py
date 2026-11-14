"""Tests for app_core/meal_log.py. Run from the repo root:  python -m pytest tests/"""

import csv
import io
from datetime import date

import pytest

from app_core.meal_log import MealLog, new_entry_id
from app_core.nutrition import calculate_nutrition

DAY1 = date(2026, 11, 16)
DAY2 = date(2026, 11, 17)


def test_add_and_list_one_meal():
    log = MealLog()
    log.add(calculate_nutrition("chicken_biryani", 150), DAY1)
    entries = log.entries_for(DAY1)
    assert len(entries) == 1
    assert entries[0]["dish"] == "chicken_biryani"


def test_totals_add_up_exact_numbers():
    log = MealLog()
    log.add(calculate_nutrition("chicken_biryani", 150), DAY1)   # 160.335 kcal
    log.add(calculate_nutrition("chapati", 100), DAY1)           # 259.0 kcal
    assert log.totals_for(DAY1)["kcal"] == pytest.approx(160.335 + 259.0)


def test_same_entry_id_is_not_added_twice():
    # This is the "double-click Add" or "Streamlit re-run" case.
    log = MealLog()
    result = calculate_nutrition("samosa", 100)
    the_id = new_entry_id()
    log.add(result, DAY1, entry_id=the_id)
    log.add(result, DAY1, entry_id=the_id)
    assert len(log.entries_for(DAY1)) == 1


def test_remove_by_id_and_totals_recompute():
    log = MealLog()
    first = log.add(calculate_nutrition("samosa", 100), DAY1)
    log.add(calculate_nutrition("chapati", 100), DAY1)
    assert log.remove(first["entry_id"]) is True
    assert len(log.entries_for(DAY1)) == 1
    assert log.totals_for(DAY1)["kcal"] == pytest.approx(259.0)


def test_remove_unknown_id_returns_false():
    assert MealLog().remove("nothing") is False


def test_dates_are_kept_separate():
    log = MealLog()
    log.add(calculate_nutrition("samosa", 100), DAY1)
    log.add(calculate_nutrition("chapati", 100), DAY2)
    assert len(log.entries_for(DAY1)) == 1
    assert log.totals_for(DAY1)["kcal"] == pytest.approx(220.0)
    assert log.totals_for(DAY2)["kcal"] == pytest.approx(259.0)


def test_date_can_be_given_as_text():
    log = MealLog()
    log.add(calculate_nutrition("daal", 100), "2026-11-16")
    assert len(log.entries_for(DAY1)) == 1


def test_bad_date_text_is_rejected():
    with pytest.raises(ValueError):
        MealLog().add(calculate_nutrition("daal", 100), "16/11/2026")


def test_empty_day_has_zero_totals():
    totals = MealLog().totals_for(DAY1)
    assert totals == {"kcal": 0.0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}


def test_csv_export_across_dates():
    log = MealLog()
    log.add(calculate_nutrition("chicken_biryani", 150), DAY1)
    log.add(calculate_nutrition("chapati", 100), DAY2)
    rows = list(csv.DictReader(io.StringIO(log.to_csv())))
    assert len(rows) == 2
    assert rows[0]["dish"] == "chicken_biryani"
    assert float(rows[0]["kcal"]) == pytest.approx(160.34, abs=0.01)   # rounded to 2 dp
    assert rows[1]["meal_date"] == "2026-11-17"


def test_csv_export_one_day_only():
    log = MealLog()
    log.add(calculate_nutrition("samosa", 100), DAY1)
    log.add(calculate_nutrition("chapati", 100), DAY2)
    rows = list(csv.DictReader(io.StringIO(log.to_csv(DAY2, all_dates=False))))
    assert len(rows) == 1 and rows[0]["dish"] == "chapati"


def test_two_logs_do_not_share_entries():
    # Like two browser sessions: each one has its own MealLog object.
    a, b = MealLog(), MealLog()
    a.add(calculate_nutrition("samosa", 100), DAY1)
    assert b.entries_for(DAY1) == []
