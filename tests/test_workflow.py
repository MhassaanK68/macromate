"""State changes, explicit confirmation, idempotency, and session isolation."""

from datetime import date

import pytest

from app_core.workflow import MealWorkflow


def ready():
    flow = MealWorkflow()
    flow.set_image(b"first photo identity")
    flow.set_predictions([{"dish": "daal", "label": "daal", "score": .5}])
    flow.confirm("chicken biryani")
    flow.calculate(150)
    return flow


def test_confirmation_is_required():
    flow = MealWorkflow()
    with pytest.raises(ValueError):
        flow.calculate(150)
    with pytest.raises(ValueError):
        flow.confirm("daal")


def test_new_image_clears_results_but_keeps_log():
    flow = ready()
    flow.add(date(2026, 11, 15))
    flow.set_image(b"second photo")
    assert not flow.suggestions and flow.confirmed_dish is None and flow.estimate is None
    assert len(flow.log.entries) == 1
    with pytest.raises(ValueError):
        flow.add(date(2026, 11, 15))


@pytest.mark.parametrize("grams", [200, 0, float("nan"), float("inf"), ""])
def test_changed_or_invalid_grams_cannot_log_old_estimate(grams):
    flow = ready()
    flow.sync_inputs("chicken_biryani", grams)
    assert flow.estimate is None
    with pytest.raises(ValueError):
        flow.add(date.today())


def test_changed_dish_requires_confirmation_again():
    flow = ready()
    flow.sync_inputs("daal", 150)
    assert flow.confirmed_dish is None and flow.estimate is None


def test_reruns_do_not_reset_current_result_or_duplicate_add():
    flow = ready()
    assert not flow.set_image(b"first photo identity")
    flow.sync_inputs("chicken_biryani", 150)
    flow.add(date.today())
    flow.add(date.today())
    assert len(flow.log.entries) == 1


def test_cancel_and_independent_sessions():
    a, b = ready(), MealWorkflow()
    a.add(date.today())
    a.cancel()
    assert a.estimate is None and a.confirmed_dish is None
    assert b.log.entries == [] and b.suggestions == []
