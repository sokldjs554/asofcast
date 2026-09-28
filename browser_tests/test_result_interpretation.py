"""The first-view comparison explains real trade-offs without using future truth."""
import pytest
from playwright.sync_api import expect
from test_choice_comparison import compare
from test_guided_flow import raw_state, ready


def test_first_view_preview_calculates_real_alternatives_without_executing_them(page):
    before = raw_state(page)
    page.locator('#previewChoices').click()
    page.wait_for_function("!document.getElementById('compareChoices').disabled")
    assert page.locator('#choiceComparison [data-status="ok"]').count() == 3
    assert float(page.locator('[data-choice="COMMIT"] .choice-forecast').get_attribute('data-value')) == before['prediction']
    assert page.locator('#revisionTimeline .revision-event').count() == 1
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0
    assert page.locator('.choice-error:visible').count() == 0
    assert page.locator('.choice-verdict:visible').count() == 0
    expect(page.locator('#comparisonInsight')).to_be_visible()


def test_tradeoff_metrics_measure_extra_wait_from_current_decision_not_origin(page):
    page.locator('#waitSelect').select_option('1800')
    ready(page)
    before = raw_state(page)
    sensor = page.locator('#compareSensor').input_value()
    cost = next(row['cost_proxy'] for row in before['candidates'] if row['sensor'] == sensor)
    compare(page)
    assert float(page.locator('[data-choice="WAIT"] .choice-wait').get_attribute('data-value')) == 1800
    assert float(page.locator('[data-choice="WAIT"] .choice-cost').get_attribute('data-value')) == 0
    assert float(page.locator('[data-choice="ACQUIRE"] .choice-wait').get_attribute('data-value')) == 0
    assert float(page.locator('[data-choice="ACQUIRE"] .choice-cost').get_attribute('data-value')) == cost


def test_audit_explains_error_change_against_commit_and_can_be_hidden_again(page):
    before = raw_state(page)
    compare(page)
    page.locator('#showComparisonTruth').check()
    baseline_error = abs(before['prediction'] - before['target_actual_retrospective'])
    for action in ('WAIT', 'ACQUIRE'):
        card = page.locator(f'[data-choice="{action}"]')
        forecast = float(card.locator('.choice-forecast').get_attribute('data-value'))
        improvement = baseline_error - abs(forecast - before['target_actual_retrospective'])
        verdict = card.locator('.choice-verdict')
        assert float(verdict.get_attribute('data-gain')) == pytest.approx(improvement)
        assert verdict.get_attribute('data-direction') == ('same' if abs(improvement) < 1e-8 else 'better' if improvement > 0 else 'worse')
        expect(verdict).to_be_visible()
    page.locator('#showComparisonTruth').uncheck()
    assert page.locator('.choice-verdict:visible').count() == 0
    assert page.locator('.choice-error:visible').count() == 0
    assert page.locator('#revisionTimeline .revision-event').count() == 1


def test_incomplete_comparison_never_names_an_overall_error_winner(page):
    page.evaluate("window.failNext = '/api/acquisition'")
    compare(page)
    page.locator('#showComparisonTruth').check()
    assert page.locator('#comparisonInsight').get_attribute('data-complete') == 'false'
    assert page.locator('#comparisonInsight').get_attribute('data-best-error') is None


def test_case_change_clears_old_interpretation_before_new_response(page):
    compare(page)
    page.locator('#showComparisonTruth').check()
    page.locator('#caseId').fill('2')
    assert page.locator('#comparisonInsight').inner_text() == ''
    expect(page.locator('#previewChoices')).to_be_disabled()


def test_action_resource_summary_uses_current_action_and_resets_on_commit(page):
    before = raw_state(page)
    sensor = page.locator('#sensorMapGrid .is-candidate .sensor-card-head strong').first.inner_text()
    cost = next(row['cost_proxy'] for row in before['candidates'] if row['sensor'] == sensor)
    page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
    ready(page)
    assert float(page.locator('#actionCost').get_attribute('data-value')) == cost
    assert float(page.locator('#actionWait').get_attribute('data-value')) == 0
    page.locator('#waitNext').click()
    ready(page)
    assert float(page.locator('#actionWait').get_attribute('data-value')) == 1800
    assert float(page.locator('#actionCost').get_attribute('data-value')) == 0
    page.locator('#commitPrediction').click()
    assert float(page.locator('#actionWait').get_attribute('data-value')) == 0
    assert float(page.locator('#actionCost').get_attribute('data-value')) == 0
