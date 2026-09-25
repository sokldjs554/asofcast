"""The first-visit path uses real forecasts; future comparisons are never user actions."""
from __future__ import annotations

from playwright.sync_api import expect


def ready(page):
    page.wait_for_function("document.getElementById('status').textContent === '' && !document.getElementById('runButton').disabled")


def raw_state(page):
    return page.evaluate("""async () => {
      const q = new URLSearchParams({case_id: document.getElementById('caseId').value,
        scenario: document.getElementById('scenario').value,
        wait_seconds: document.getElementById('waitSelect').value});
      return JSON.parse((await asgiFetch('/api/acquisition?' + q, 'GET', null)).body);
    }""")


def choose_real_action(page, action):
    """Locate a real case, never force a recommendation or alter a model score."""
    found = page.evaluate("""async (action) => {
      const meta = JSON.parse((await asgiFetch('/api/metadata', 'GET', null)).body);
      for (const wait of meta.config.waits_seconds) {
        for (let id = 0; id < Math.min(128, meta.cases); id++) {
          const q = new URLSearchParams({case_id:id,scenario:'mixed',wait_seconds:wait});
          const r = JSON.parse((await asgiFetch('/api/acquisition?' + q,'GET',null)).body);
          if (r.recommended_action === action) return {id,wait};
        }
      }
      return null;
    }""", action)
    assert found is not None, f'actual model fixture must contain {action}'
    page.locator('#caseId').fill(str(found['id']))
    page.locator('#waitSelect').select_option(str(found['wait']))
    page.locator('#runButton').click()
    ready(page)


def test_first_visit_names_target_horizon_source_and_next_action(page):
    expect(page.locator('h1')).to_contain_text('센서가 늦게 도착할 때')
    s = raw_state(page)
    meta = page.evaluate("async () => JSON.parse((await asgiFetch('/api/metadata','GET',null)).body)")
    assert meta['config']['target'] in page.locator('#forecastSubject').inner_text()
    assert page.locator('#targetTime').get_attribute('data-timestamp') == str(s['target_time'])
    assert page.locator('#originTime').get_attribute('data-timestamp') == str(s['origin_time'])
    assert page.locator('#missingSensors').inner_text() == str(s['total_sensors'] - s['available_origin_sensors'])
    assert '실제' in page.locator('#actionHelp').inner_text()
    assert page.locator('#technicalDetails').get_attribute('open') is None
    expect(page.locator('#followRecommendation')).to_be_enabled()
    assert '선택 전' in page.locator('#sessionStatus').inner_text()
    assert page.locator('#revisionTimeline .revision-event').count() == 1


def test_wait_button_advances_replay_without_moving_target_or_fabricating_acquisition(page):
    assert page.locator('#followRecommendation').count() == 1
    choose_real_action(page, 'WAIT')
    before = raw_state(page)
    target = page.locator('#targetTime').get_attribute('data-timestamp')
    assert '기다리지' in page.locator('#actionHelp').inner_text()
    page.locator('#followRecommendation').click()
    ready(page)
    assert float(page.locator('#waitSelect').input_value()) == before['next_wait_seconds']
    assert page.locator('#targetTime').get_attribute('data-timestamp') == target
    assert page.locator('#sensorMapGrid .is-acquired').count() == 0
    assert '대기 재생' in page.locator('#revisionTimeline').inner_text()
    assert '모델 추천' in page.locator('#resultExplanation').inner_text()
    assert '정확도 향상' in page.locator('#resultCaution').inner_text()


def test_manual_sensor_choice_records_actual_before_after_and_actor(page):
    assert page.locator('#beforeForecast').count() == 1
    before = page.locator('#forecast').inner_text()
    target = page.locator('#targetTime').get_attribute('data-timestamp')
    card = page.locator('#sensorMapGrid .sensor-card.is-candidate').first
    sensor = card.locator('.sensor-card-head strong').inner_text()
    card.locator('.sensor-acquire').click()
    ready(page)
    assert page.locator('#beforeForecast').inner_text() == before
    assert page.locator('#afterForecast').inner_text() == page.locator('#forecast').inner_text()
    assert page.locator('#targetTime').get_attribute('data-timestamp') == target
    assert '직접 선택' in page.locator('#resultExplanation').inner_text()
    assert sensor in page.locator('#resultExplanation').inner_text()
    assert page.locator('#revisionTimeline .revision-event').count() == 2
    assert page.locator('#resultEvaluation').get_attribute('open') is None


def test_manual_commit_is_local_and_reset_starts_a_new_session(page):
    assert page.locator('#commitPrediction').count() == 1
    before = page.locator('#forecast').inner_text()
    requests_before = page.evaluate('traffic.length')
    page.locator('#commitPrediction').click()
    assert page.locator('#afterForecast').inner_text() == before
    assert '화면에서 확정' in page.locator('#sessionStatus').inner_text()
    assert '서버에 저장' in page.locator('#resultExplanation').inner_text()
    assert page.evaluate('traffic.length') == requests_before
    expect(page.locator('#followRecommendation')).to_be_disabled()
    assert page.locator('#sensorMapGrid .sensor-acquire:enabled').count() == 0
    page.locator('#resetAcquisition').click()
    ready(page)
    assert page.locator('#waitSelect').input_value() == '0'
    assert '선택 전' in page.locator('#sessionStatus').inner_text()
    assert page.locator('#revisionTimeline .revision-event').count() == 1


def test_recommended_commit_does_not_claim_a_manual_or_remote_action(page):
    assert page.locator('#followRecommendation').count() == 1
    choose_real_action(page, 'COMMIT')
    page.locator('#followRecommendation').click()
    assert '모델 추천' in page.locator('#resultExplanation').inner_text()
    assert '화면에서 확정' in page.locator('#sessionStatus').inner_text()


def test_old_acquisition_cannot_restore_previous_result_after_case_change(page):
    assert page.locator('#afterForecast').count() == 1
    page.evaluate("window.holdNext = {path:'/api/acquire'}")
    page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    expected = page.locator('#afterForecast').inner_text()
    page.evaluate('window.held.release()')
    page.wait_for_function('window.held === null')
    page.wait_for_timeout(150)
    assert page.locator('#afterForecast').inner_text() == expected
    assert '선택 전' in page.locator('#sessionStatus').inner_text()
    assert page.locator('#revisionTimeline .revision-event').count() == 1


def test_real_models_and_evidence_are_accessible_inside_the_same_page(page):
    assert page.locator('#technicalDetails').count() == 1
    page.locator('#technicalDetails > summary').click()
    expect(page.locator('#counterfactual')).to_be_visible()
    assert page.locator('#counterfactualRows tr').count() == 7
    assert page.locator('#paretoChart circle').count() == 5
    assert '별도 비교' in page.locator('#comparisonHeading').inner_text()
    assert '실행한 기록이 아닙니다' in page.locator('#comparisonScope').inner_text()
    page.set_viewport_size({'width': 390, 'height': 844})
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')


def test_primary_choice_is_visible_in_desktop_first_viewport(page):
    box = page.locator('#followRecommendation').bounding_box()
    assert box and box['y'] + box['height'] <= 1000, 'the introduction must not bury the first action'


def test_recommended_sensor_is_not_recorded_as_a_manual_choice(page):
    choose_real_action(page, 'ACQUIRE')
    before = raw_state(page)
    sensor = before['recommended_sensor']
    page.locator('#followRecommendation').click()
    ready(page)
    assert '모델 추천' in page.locator('#resultExplanation').inner_text()
    assert sensor in page.locator('#resultExplanation').inner_text()
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1


def test_wait_keeps_previously_read_sensors_and_fixed_target(page):
    card = page.locator('#sensorMapGrid .sensor-card.is-candidate').first
    sensor = card.locator('.sensor-card-head strong').inner_text()
    card.locator('.sensor-acquire').click()
    ready(page)
    target = page.locator('#targetTime').get_attribute('data-timestamp')
    page.locator('#waitNext').click()
    ready(page)
    last = page.evaluate("traffic.filter(r => r.path === '/api/acquisition').slice(-1)[0]")
    assert page.locator('#targetTime').get_attribute('data-timestamp') == target
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1
    assert sensor in page.locator('#revisionTimeline').inner_text()
    assert '대기 재생' in page.locator('#revisionTimeline').inner_text()
    assert last['status'] == 200


def test_comparison_failure_cannot_turn_successful_sensor_read_into_failure(page):
    page.evaluate("window.failNext = '/api/revision-timeline'")
    page.locator('#sensorMapGrid .is-candidate .sensor-acquire').first.click()
    ready(page)
    assert page.locator('#sensorMapGrid .is-acquired').count() == 1
    assert '직접 선택' in page.locator('#resultExplanation').inner_text()
    page.locator('#technicalDetails > summary').click()
    assert '불러오지 못했습니다' in page.locator('#comparisonTimeline').inner_text()


def test_narrow_screen_and_keyboard_case_selection(page):
    page.set_viewport_size({'width': 320, 'height': 740})
    assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
    page.locator('#caseId').fill('2')
    expect(page.locator('#followRecommendation')).to_be_disabled()
    assert page.locator('#afterForecast').inner_text() == '—'
    page.locator('#caseId').press('Enter')
    ready(page)
    assert page.locator('#caseId').input_value() == '2'
    assert page.locator('#revisionTimeline .revision-event').count() == 1
    expect(page.locator('#followRecommendation')).to_be_enabled()


def test_delayed_wait_cannot_overwrite_new_case_result(page):
    page.evaluate("window.holdNext = {path:'/api/acquisition', caseId:'0'}")
    page.locator('#waitNext').click()
    page.wait_for_function('window.held !== null')
    page.locator('#nextCase').click()
    ready(page)
    target = page.locator('#targetTime').get_attribute('data-timestamp')
    expected = page.locator('#beforeForecast').inner_text()
    page.evaluate('window.held.release()')
    page.wait_for_function('window.held === null')
    page.wait_for_timeout(150)
    assert page.locator('#beforeForecast').inner_text() == expected
    assert page.locator('#targetTime').get_attribute('data-timestamp') == target
    assert '선택 전' in page.locator('#sessionStatus').inner_text()
    assert page.locator('#revisionTimeline .revision-event').count() == 1


def test_last_decision_time_cannot_advance_beyond_trained_grid(page):
    last = page.evaluate("async () => JSON.parse((await asgiFetch('/api/metadata','GET',null)).body).config.waits_seconds.slice(-1)[0]")
    page.locator('#waitSelect').select_option(str(int(last)))
    ready(page)
    expect(page.locator('#waitNext')).to_be_disabled()
    assert '마지막' in page.locator('#waitNext').inner_text()
    expect(page.locator('#commitPrediction')).to_be_enabled()
