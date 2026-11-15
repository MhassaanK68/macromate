"""Nutrition calculator for MacroMate.

Job of this file:
    confirmed dish name + serving grams  ->  kcal, protein, carbs, fat

The numbers come from data/nutrition.csv, which stores values PER 100 GRAMS.
The formula is:

    amount = value_per_100g * (grams / 100)

Example: chicken_biryani has 106.89 kcal per 100 g.
         For 150 g:  106.89 * (150 / 100) = 160.335 kcal
"""

import csv
import math
from pathlib import Path

# Where the CSV file lives (one folder up from app_core/, then data/).
NUTRITION_CSV = Path(__file__).resolve().parent.parent / "data" / "nutrition.csv"

# A safety limit so nobody types 10000000 by mistake. This is OUR choice, not
# an assignment rule.
MAX_GRAMS = 5000
SUPPORTED_LABELS = {"chicken_biryani", "chicken_pulao", "chicken_karahi", "daal",
                    "nihari", "haleem", "samosa", "chapati"}

# The CSV columns we need, and the names we give them in our results.
NUTRIENT_COLUMNS = {
    "kcal": "kcal_per_100g",
    "protein_g": "protein_g_per_100g",
    "carbs_g": "carbs_g_per_100g",
    "fat_g": "fat_g_per_100g",
}


class NutritionError(ValueError):
    """Raised when the dish or grams are not valid.

    The app can catch this and show the message to the user.
    """


def clean_label(label):
    """Make dish names match.

    The model says "chicken biryani" (with a space) but the CSV says
    "chicken_biryani" (with an underscore). We turn both into the CSV style.
    """
    return str(label).strip().lower().replace(" ", "_")


def _read_number(text, column, dish):
    """Turn CSV text into a number. Reject anything missing, negative or weird."""
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise NutritionError(f"Nutrition data for {dish}: '{column}' is not a number")
    if not math.isfinite(number) or number < 0:
        raise NutritionError(f"Nutrition data for {dish}: '{column}' must be 0 or more")
    return number


def load_nutrition_table(csv_path=NUTRITION_CSV):
    """Read the CSV and return a dictionary: dish name -> its row of data.

    Example result:
        {"chicken_biryani": {"kcal": 106.89, "protein_g": 6.63, ...}, ...}
    """
    table = {}
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"dish_label", "source", "source_id", "recipe_assumption", "quality_flag",
                    *NUTRIENT_COLUMNS.values()}
        if not required.issubset(reader.fieldnames or []):
            raise NutritionError("Nutrition CSV is missing required columns")
        for row in reader:
            dish = clean_label(row["dish_label"])
            if dish in table:
                raise NutritionError(f"Dish '{dish}' appears twice in the nutrition CSV")
            entry = {}
            for name, column in NUTRIENT_COLUMNS.items():
                entry[name] = _read_number(row[column], column, dish)
            # Extra information we show to the user so they know it is an estimate.
            entry["source"] = row["source"]
            entry["source_id"] = row["source_id"]
            entry["recipe_assumption"] = row["recipe_assumption"]
            entry["quality_flag"] = row["quality_flag"]
            table[dish] = entry
    if set(table) != SUPPORTED_LABELS:
        raise NutritionError("Nutrition CSV must contain exactly the eight supported dishes")
    return table


# We load the CSV only once and remember it here.
_table_cache = None


def get_table():
    """Return the nutrition table, loading it the first time only."""
    global _table_cache
    if _table_cache is None:
        _table_cache = load_nutrition_table()
    return _table_cache


def check_grams(grams):
    """Make sure grams is a real number between 0 and MAX_GRAMS. Return it as a float."""
    # True/False are numbers in Python (True == 1), so block them on purpose.
    if isinstance(grams, bool):
        raise NutritionError("Serving grams must be a number")
    try:
        grams = float(grams)
    except (TypeError, ValueError):
        raise NutritionError("Serving grams must be a number")
    if not math.isfinite(grams):          # blocks nan and infinity
        raise NutritionError("Serving grams must be a normal number")
    if grams <= 0:
        raise NutritionError("Serving grams must be greater than 0")
    if grams > MAX_GRAMS:
        raise NutritionError(f"Serving grams must be {MAX_GRAMS} or less")
    return grams


def calculate_nutrition(label, grams, table=None):
    """Calculate the nutrition for one confirmed dish and serving size.

    label: dish name, with a space or underscore ("chicken biryani")
    grams: serving size typed by the user
    table: optional, only used in tests. Normally we use the real CSV.

    Returns a dictionary with kcal, protein_g, carbs_g, fat_g (NOT rounded)
    plus the source notes. Round only when you DISPLAY the numbers.
    """
    if table is None:
        table = get_table()

    dish = clean_label(label)
    if dish not in table:
        # We never guess the "nearest" dish. Unsupported means no numbers.
        raise NutritionError(f"'{label}' is not a supported dish")

    grams = check_grams(grams)
    reference = table[dish]
    factor = grams / 100.0     # the CSV is per 100 g

    return {
        "dish": dish,
        "serving_grams": grams,
        "kcal": reference["kcal"] * factor,
        "protein_g": reference["protein_g"] * factor,
        "carbs_g": reference["carbs_g"] * factor,
        "fat_g": reference["fat_g"] * factor,
        "source": reference["source"],
        "source_id": reference["source_id"],
        "recipe_assumption": reference["recipe_assumption"],
        "quality_flag": reference["quality_flag"],
    }
