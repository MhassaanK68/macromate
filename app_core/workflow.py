"""Per-session workflow; UI reruns never silently reuse stale estimates."""

from dataclasses import dataclass, field
from hashlib import sha256

from .meal_log import MealLog, new_entry_id
from .nutrition import NutritionError, calculate_nutrition, check_grams, clean_label


@dataclass
class MealWorkflow:
    image_id: str | None = None
    suggestions: list = field(default_factory=list)
    confirmed_dish: str | None = None
    estimate: dict | None = None
    entry_id: str | None = None
    added: bool = False
    log: MealLog = field(default_factory=MealLog)
    removed_entry: tuple | None = None

    def set_image(self, content):
        identity = sha256(content).hexdigest() if content else None
        if identity == self.image_id:
            return False
        self.image_id = identity
        self.suggestions = []
        self.cancel()
        return True

    def set_predictions(self, suggestions):
        if self.image_id is None:
            raise ValueError("Upload a photo first")
        self.cancel()
        self.suggestions = list(suggestions)

    def cancel(self):
        self.confirmed_dish = None
        self.estimate = None
        self.entry_id = None
        self.added = False

    def confirm(self, dish):
        if not self.suggestions:
            raise ValueError("Predict a dish first")
        self.cancel()
        calculate_nutrition(dish, 100)
        self.confirmed_dish = clean_label(dish)

    def sync_inputs(self, dish, grams):
        if self.confirmed_dish and clean_label(dish) != self.confirmed_dish:
            self.cancel()
        if self.estimate:
            try:
                unchanged = check_grams(grams) == self.estimate["serving_grams"]
            except NutritionError:
                unchanged = False
            if not unchanged:
                self.estimate = None
                self.entry_id = None
                self.added = False

    def calculate(self, grams):
        if not self.confirmed_dish:
            raise ValueError("Confirm a supported dish first")
        self.estimate = calculate_nutrition(self.confirmed_dish, grams)
        self.entry_id = new_entry_id()
        self.added = False
        return self.estimate

    def add(self, meal_date):
        if self.estimate is None or self.entry_id is None:
            raise ValueError("Calculate the current serving first")
        entry = self.log.add(self.estimate, meal_date, entry_id=self.entry_id)
        self.added = True
        return entry

    def remove(self, entry_id):
        """Keep the last removed entry so a mistaken tap can be undone."""
        for index, entry in enumerate(self.log.entries):
            if entry["entry_id"] == entry_id:
                self.removed_entry = (index, dict(entry))
                self.log.remove(entry_id)
                if self.entry_id == entry_id:
                    self.cancel()
                return True
        return False

    def undo_remove(self):
        if self.removed_entry is None:
            return False
        index, entry = self.removed_entry
        if not any(item["entry_id"] == entry["entry_id"] for item in self.log.entries):
            self.log.entries.insert(min(index, len(self.log.entries)), entry)
        self.removed_entry = None
        return True
