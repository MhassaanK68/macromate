"""Daily meal log for MacroMate.

The log lives in memory only. It is a temporary list that disappears when the
app session ends, so the app should also offer CSV export (to_csv below).

Each entry is a small dictionary:
    entry_id, meal_date, added_at, dish, serving_grams,
    kcal, protein_g, carbs_g, fat_g, source_id
"""

import csv
import io
import uuid
from datetime import date, datetime

# Column order for the CSV export.
CSV_COLUMNS = ["entry_id", "meal_date", "added_at", "dish", "serving_grams",
               "kcal", "protein_g", "carbs_g", "fat_g", "source_id"]

NUMBER_FIELDS = ["kcal", "protein_g", "carbs_g", "fat_g"]


def new_entry_id():
    """Make a unique ID. The UI should make ONE ID per calculated result.

    Why: Streamlit re-runs the script on every click. If we made a new ID on
    every re-run, a double-click would add the meal twice. With one ID per
    result, the second add is recognised and ignored.
    """
    return uuid.uuid4().hex


def to_date(value):
    """Accept a date, a datetime, or text like '2026-11-16'. Return a date."""
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Meal date must look like 2026-11-16, got '{value}'")


class MealLog:
    def __init__(self):
        self.entries = []          # list of entry dictionaries

    def add(self, nutrition, meal_date=None, entry_id=None):
        """Add one meal.

        nutrition: the dictionary returned by calculate_nutrition()
        meal_date: the day the user ate it (date or 'YYYY-MM-DD'); default today
        entry_id:  pass the ID made by new_entry_id() to stop duplicates

        Returns the entry. If this entry_id is already in the log, nothing is
        added and the existing entry is returned.
        """
        if entry_id is None:
            entry_id = new_entry_id()

        for existing in self.entries:
            if existing["entry_id"] == entry_id:
                return existing            # already added, do not duplicate

        entry = {
            "entry_id": entry_id,
            "meal_date": to_date(meal_date),
            "added_at": datetime.now().isoformat(timespec="seconds"),
            "dish": nutrition["dish"],
            "serving_grams": nutrition["serving_grams"],
            "kcal": nutrition["kcal"],
            "protein_g": nutrition["protein_g"],
            "carbs_g": nutrition["carbs_g"],
            "fat_g": nutrition["fat_g"],
            "source_id": nutrition["source_id"],
        }
        self.entries.append(entry)
        return entry

    def remove(self, entry_id):
        """Delete an entry by ID. Return True if it was found, else False."""
        for entry in self.entries:
            if entry["entry_id"] == entry_id:
                self.entries.remove(entry)
                return True
        return False

    def entries_for(self, meal_date=None):
        """All entries for one day (default today), oldest first."""
        day = to_date(meal_date)
        return [e for e in self.entries if e["meal_date"] == day]

    def totals_for(self, meal_date=None):
        """Add up kcal, protein, carbs and fat for one day.

        We add the exact (unrounded) numbers. Round only when showing them.
        """
        totals = {name: 0.0 for name in NUMBER_FIELDS}
        for entry in self.entries_for(meal_date):
            for name in NUMBER_FIELDS:
                totals[name] += entry[name]
        return totals

    def to_csv(self, meal_date=None, all_dates=True):
        """Return the log as CSV text.

        By default every date is exported. Pass all_dates=False together with
        meal_date to export just one day. Numbers are rounded to 2 decimals
        here because the file is for people to read.
        """
        if all_dates:
            rows = self.entries
        else:
            rows = self.entries_for(meal_date)

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for entry in rows:
            row = dict(entry)
            row["serving_grams"] = round(row["serving_grams"], 2)
            for name in NUMBER_FIELDS:
                row[name] = round(row[name], 2)
            writer.writerow(row)
        return output.getvalue()
