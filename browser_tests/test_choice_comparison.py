"""Independent alternatives must never execute a session choice or leak stale results."""
import json

from playwright.sync_api import expect
from test_guided_flow import raw_state, ready


def compare(page):
    page.locator('#compareChoices').click()
    page.wait_for_function("!document.getElementById('compareChoices').disabled")


def test_alternatives_use_real_models_from_one_start_without_changing_session(page):
    before = raw_state(page)
    sensor = page.locator('#compareSensor').input_value()
    compare(page)
    rows = page.locator('#choiceComparison .choice-card')
    assert rows.count() == 3
    assert page.locator('#forecast').inner_text() == f"{before['prediction']:.2f}"
    assert page.locator('#revisionTimeline .revision-event').count() == 1
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0
    acquired = page.evaluate("""async (sensor) => JSON.parse((await asgiFetch('/api/acquire', 'POST',
      JSON.stringify({case_id:0,scenario:'mixed',wait_seconds:0,acquired:[],sensor}))).body)""", sensor)
    actual = float(page.locator('[data-choice="ACQUIRE"] .choice-forecast').get_attribute('data-value'))
    assert abs(actual - acquired['prediction']) < 1e-7
    assert page.locator('[data-choice="ACQUIRE"]').get_attribute('data-target') == str(before['target_time'])
    assert page.locator('[data-choice="WAIT"]').get_attribute('data-target') == str(before['target_time'])
    assert page.locator('.choice-error:visible').count() == 0


def test_audit_is_opt_in_and_does_not_change_recommendation(page):
    action = page.locator('#actionBadge').inner_text()
    compare(page)
    page.locator('#showComparisonTruth').check()
    assert page.locator('.choice-error:visible').count() == 3
    assert page.locator('#actionBadge').inner_text() == action
    assert page.locator('#revisionTimeline .revision-event').count() == 1
    before = raw_state(page)
    actual_error = float(page.locator('[data-choice="COMMIT"] .choice-error').get_attribute('data-value'))
    assert abs(actual_error - abs(before['prediction'] - before['target_actual_retrospective'])) < 1e-7


def test_delayed_comparison_cannot_restore_old_case(page):
    page.evaluate("window.holdNext = {path:'/api/acquisition',caseId:'0'}")
    page.locator('#compareChoices').click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    page.evaluate('window.held.release()')
    page.wait_for_function('window.held === null')
    page.wait_for_timeout(150)
    assert page.locator('#choiceComparison .choice-card').count() == 0
    expect(page.locator('#exportComparison')).to_be_disabled()
    assert page.locator('#caseId').input_value() == '1'


def test_partial_failure_preserves_other_choices_and_retry(page):
    page.evaluate("window.failNext = '/api/acquisition'")
    compare(page)
    assert page.locator('[data-choice="WAIT"]').get_attribute('data-status') == 'error'
    assert '계산할 대기 선택' in page.locator('[data-choice="WAIT"] .choice-condition').inner_text()
    assert page.locator('[data-choice="ACQUIRE"]').get_attribute('data-status') == 'ok'
    expect(page.locator('#followRecommendation')).to_be_enabled()
    compare(page)
    assert page.locator('[data-choice="WAIT"]').get_attribute('data-status') == 'ok'


def test_final_grid_has_no_future_wait(page):
    page.locator('#waitSelect').select_option('3600')
    ready(page)
    compare(page)
    assert page.locator('[data-choice="WAIT"]').get_attribute('data-status') == 'unavailable'
    assert '—' in page.locator('[data-choice="WAIT"] .choice-forecast').inner_text()


def test_export_keeps_provenance_and_truth_hidden_until_requested(page, tmp_path):
    compare(page)
    with page.expect_download() as download:
        page.locator('#exportComparison').click()
    path = tmp_path / 'comparison.json'
    download.value.save_as(path)
    evidence = json.loads(path.read_text())
    assert evidence['source_kind'] == raw_state(page)['source_kind']
    assert evidence['run_id']
    assert evidence['scope'] == 'single_case_counterfactual_not_executed'
    assert 'target_actual_retrospective' not in evidence
    assert all('absolute_error_retrospective' not in row for row in evidence['choices'])
    assert evidence['start']['case_id'] == 0
    assert len(evidence['choices']) == 3


def test_comparison_preserves_selected_sensor_when_committing(page):
    options = page.locator('#compareSensor option').evaluate_all('(options) => options.map(option => option.value)')
    assert len(options) >= 2
    sensor = options[1]
    page.locator('#compareSensor').select_option(sensor)
    compare(page)
    page.locator('#commitPrediction').click()
    assert page.locator('#compareSensor').input_value() == sensor
    assert sensor in page.locator('[data-choice="ACQUIRE"] .choice-condition').inner_text()


def test_commit_during_comparison_keeps_the_selected_sensor(page):
    sensor = page.locator('#compareSensor option').nth(1).get_attribute('value')
    page.locator('#compareSensor').select_option(sensor)
    page.evaluate("window.holdNext = {path:'/api/acquisition',caseId:'0'}")
    page.locator('#compareChoices').click()
    page.wait_for_function('window.held !== null')
    page.locator('#commitPrediction').click()
    page.evaluate('window.held.release()')
    page.wait_for_function("!document.getElementById('compareChoices').disabled")
    assert page.locator('#compareSensor').input_value() == sensor
    assert sensor in page.locator('[data-choice="ACQUIRE"] .choice-condition').inner_text()


def test_wait_comparison_keeps_already_acquired_origin_values(page):
    card = page.locator('#sensorMapGrid .is-candidate').first
    sensor = card.locator('.sensor-card-head strong').inner_text()
    card.locator('.sensor-acquire').click()
    ready(page)
    next_sensor = page.locator('#compareSensor').input_value()
    compare(page)
    expected = page.evaluate("""async (sensor) => JSON.parse((await asgiFetch(
      '/api/acquisition?case_id=0&scenario=mixed&wait_seconds=1800&acquired=' + sensor,
      'GET', null)).body)""", sensor)
    forecast = float(page.locator('[data-choice="WAIT"] .choice-forecast').get_attribute('data-value'))
    assert abs(forecast - expected['prediction']) < 1e-7
    expected_acquire = page.evaluate("""async ([sensor, nextSensor]) => JSON.parse((await asgiFetch(
      '/api/acquire', 'POST', JSON.stringify({case_id:0,scenario:'mixed',wait_seconds:0,
      acquired:[sensor],sensor:nextSensor}))).body)""", [sensor, next_sensor])
    forecast_acquire = float(page.locator('[data-choice="ACQUIRE"] .choice-forecast').get_attribute('data-value'))
    assert abs(forecast_acquire - expected_acquire['prediction']) < 1e-7
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1


def test_all_sensors_received_does_not_invent_an_additional_read(page):
    while page.locator('#sensorMapGrid .is-candidate').count():
        page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
        ready(page)
    expect(page.locator('#compareSensor')).to_be_disabled()
    compare(page)
    assert page.locator('[data-choice="ACQUIRE"]').get_attribute('data-status') == 'unavailable'
    assert '—' in page.locator('[data-choice="ACQUIRE"] .choice-forecast').inner_text()
