"""Tests for app_core/nutrition.py. Run from the repo root:  python -m pytest tests/"""

import math

import pytest

from app_core.nutrition import (NutritionError, calculate_nutrition, clean_label,
                                get_table)

# The eight dishes the model can recognise (spelled the way the MODEL spells them).
MODEL_LABELS = ["chicken biryani", "chicken pulao", "chicken karahi", "daal",
                "nihari", "haleem", "samosa", "chapati"]


def test_biryani_150_grams():
    # 106.89 kcal per 100 g * 1.5 = 160.335
    result = calculate_nutrition("chicken_biryani", 150)
    assert result["kcal"] == pytest.approx(160.335)
    assert result["protein_g"] == pytest.approx(9.945)
    assert result["carbs_g"] == pytest.approx(21.48)
    assert result["fat_g"] == pytest.approx(3.135)


def test_100_grams_gives_the_csv_values():
    result = calculate_nutrition("chicken_biryani", 100)
    assert result["kcal"] == pytest.approx(106.89)


def test_fractional_serving():
    # 37.5 g is a quarter of 150 g, so every number is a quarter too.
    full = calculate_nutrition("samosa", 150)
    part = calculate_nutrition("samosa", 37.5)
    assert part["kcal"] == pytest.approx(full["kcal"] / 4)


def test_label_with_space_works():
    # The model says "chicken biryani", the CSV says "chicken_biryani".
    assert calculate_nutrition("chicken biryani", 100)["dish"] == "chicken_biryani"
    assert clean_label("  Chicken Karahi ") == "chicken_karahi"


def test_all_eight_model_labels_have_exactly_one_row():
    table = get_table()
    assert len(table) == 8
    for label in MODEL_LABELS:
        assert clean_label(label) in table
        assert calculate_nutrition(label, 100)["kcal"] > 0


def test_grams_given_as_text_is_accepted():
    assert calculate_nutrition("daal", "200")["serving_grams"] == 200.0


@pytest.mark.parametrize("bad", [0, -5, -0.1, "abc", "", None, math.nan, math.inf,
                                 -math.inf, True, 5001])
def test_bad_grams_are_rejected(bad):
    with pytest.raises(NutritionError):
        calculate_nutrition("chicken_biryani", bad)


def test_unsupported_dish_is_rejected():
    with pytest.raises(NutritionError):
        calculate_nutrition("pizza", 100)


def test_result_contains_source_notes():
    result = calculate_nutrition("nihari", 100)
    for key in ("source", "source_id", "recipe_assumption", "quality_flag"):
        assert result[key]


def test_bad_csv_value_is_rejected(tmp_path):
    from app_core.nutrition import load_nutrition_table
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(
        "dish_label,kcal_per_100g,protein_g_per_100g,carbs_g_per_100g,fat_g_per_100g,"
        "source,source_id,recipe_assumption,quality_flag\n"
        "daal,-1,7,21,6,src,N1,note,flag\n", encoding="utf-8")
    with pytest.raises(NutritionError):
        load_nutrition_table(bad_csv)
