"""Browser regressions against real FastAPI/PyTorch responses, not fabricated scores."""

import pytest


def ready(page):
    page.wait_for_function("document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled")


def hold(page, path, case_id=None):
    value = {'path': path}
    if case_id is not None:
        value['caseId'] = str(case_id)
    page.evaluate('(value) => window.holdNext = value', value)


def release(page):
    page.evaluate('window.held.release()')
    page.wait_for_function('window.held === null')
    # Drain response.json(), rendering and the next bridge response if scheduled.
    page.wait_for_timeout(200)


def first_sensor(page):
    return page.locator('#sensorMapGrid .sensor-card.is-candidate .sensor-acquire').first


def test_delayed_acquire_cannot_overwrite_new_case(page):
    hold(page, '/api/acquire')
    first_sensor(page).click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    expected = page.locator('#forecast').inner_text()
    release(page)
    assert page.locator('#caseId').input_value() == '1'
    assert page.locator('#forecast').inner_text() == expected
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0
    assert '취득 ·' not in page.locator('#revisionTimeline').inner_text()


def test_delayed_acquire_cannot_undo_reset(page):
    original = page.locator('#forecast').inner_text()
    hold(page, '/api/acquire')
    first_sensor(page).click()
    page.wait_for_function('window.held !== null')
    page.locator('#resetAcquisition').click()
    ready(page)
    release(page)
    assert page.locator('#forecast').inner_text() == original
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0


def test_delayed_timeline_cannot_overwrite_new_case(page):
    hold(page, '/api/revision-timeline', 1)
    page.locator('#nextCase').click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    expected = page.locator('#revisionTimeline').inner_text()
    release(page)
    assert page.locator('#caseId').input_value() == '2'
    assert page.locator('#revisionTimeline').inner_text() == expected


def test_overlapping_sensor_clicks_issue_only_one_pull(page):
    hold(page, '/api/acquire')
    page.evaluate("""() => {
      const buttons = [...document.querySelectorAll('#sensorMapGrid .is-candidate .sensor-acquire')];
      if (buttons.length < 2) throw Error('fixture requires two candidates');
      buttons[0].click(); buttons[1].click();
    }""")
    page.wait_for_function('window.held !== null')
    assert page.evaluate("traffic.filter(r => r.path === '/api/acquire').length") == 1
    release(page)
    ready(page)
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1
    first_sensor(page).click()
    ready(page)
    assert page.locator('#sensorMapGrid .is-acquired').count() == 2


def test_unapplied_case_input_cannot_use_old_sensor_state(page):
    page.locator('#caseId').fill('2')
    # Native button activation must be ignored, or the UI must first reload the case.
    page.evaluate("document.querySelector('#sensorMapGrid .is-candidate .sensor-acquire').click()")
    page.wait_for_timeout(200)
    assert page.evaluate("traffic.filter(r => r.path === '/api/acquire').length") == 0


def test_unknown_case_does_not_leave_acquisition_enabled(page):
    page.locator('#caseId').fill('99999')
    page.locator('#runButton').click()
    page.wait_for_function("document.getElementById('status').classList.contains('error')")
    assert page.locator('#sensorMapGrid .sensor-acquire:enabled').count() == 0


@pytest.mark.parametrize(('control', 'value'), [('scenario', 'outage'), ('waitSelect', '1800')])
def test_delayed_acquire_cannot_overwrite_changed_controls(page, control, value):
    hold(page, '/api/acquire')
    first_sensor(page).click()
    page.wait_for_function('window.held !== null')
    page.locator('#' + control).select_option(value)
    ready(page)
    expected = page.locator('#forecast').inner_text()
    release(page)
    assert page.locator('#forecast').inner_text() == expected
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0


def test_failed_acquire_releases_lock_for_retry(page):
    page.evaluate("window.failNext = '/api/acquire'")
    first_sensor(page).click()
    page.wait_for_function("document.getElementById('status').classList.contains('error')")
    assert page.locator('#sensorMapGrid .sensor-acquire:enabled').count() > 0
    first_sensor(page).click()
    ready(page)
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1


def test_stale_error_cannot_replace_new_case_success(page):
    hold(page, '/api/acquire')
    page.evaluate("window.failNext = '/api/acquire'")
    first_sensor(page).click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    release(page)
    assert page.locator('#status').inner_text() == ''
    assert page.locator('#caseId').input_value() == '1'
