'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const put = (id, value) => { $(id).textContent = value; };
  const fmt = (value, digits = 2) => Number.isFinite(value)
    ? value.toLocaleString('ko-KR', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—';
  const stamp = seconds => new Date(seconds * 1000).toISOString().slice(0, 16).replace('T', ' ');
  let metadata = null, current = null, controller = null, requestNumber = 0;
  const methods = {
    arrival_learned: '학습 모델 · 학습한 대기 판단',
    arrival_immediate: '학습 모델 · 즉시 확정',
    arrival_fixed_1800s: '학습 모델 · 30분 고정 대기',
    arrival_deadline: '학습 모델 · 60분 고정 대기',
    arrival_random_matched: '같은 대기 분포 · 무작위 기대값',
    value_only_immediate: '동일 크기 값 전용 모델 · 즉시',
    value_only_fixed_1800s: '동일 크기 값 전용 모델 · 30분',
    value_only_deadline: '동일 크기 값 전용 모델 · 60분',
    dlinear_immediate: '분해·선형 기준 모델 · 즉시',
    dlinear_fixed_1800s: '분해·선형 기준 모델 · 30분',
    dlinear_deadline: '분해·선형 기준 모델 · 60분'
  };
  function status(message, error = false) {
    put('status', message); $('status').classList.toggle('error', error);
  }
  async function getJSON(url, signal) {
    const response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(`${response.status}: ${error.detail || '요청을 처리하지 못했습니다.'}`);
    }
    return response.json();
  }
  function cell(row, value) {
    const td = document.createElement('td'); td.textContent = value; row.append(td); return td;
  }
  function evidence(scenario) {
    const metrics = scenario === 'outage' ? metadata.outage_test_metrics : metadata.test_metrics;
    const body = $('results'); body.replaceChildren();
    Object.keys(methods).filter(key => metrics[key]).forEach(key => {
      const m = metrics[key], row = document.createElement('tr');
      if (key === 'arrival_learned') row.className = 'metric-highlight';
      cell(row, methods[key]); cell(row, fmt(m.mae, 4));
      cell(row, `${fmt(m.mean_wait_seconds / 60, 1)}분`); cell(row, `${m.n}건`);
      body.append(row);
    });
  }
  function drawChart(active) {
    const svg = $('chart'), NS = 'http://www.w3.org/2000/svg'; svg.replaceChildren();
    const points = current.steps, values = [current.target_actual];
    points.forEach(s => values.push(s.prediction, s.baseline_prediction));
    const min = Math.min(...values), max = Math.max(...values), margin = Math.max((max - min) * .35, .25);
    const low = min - margin, high = max + margin;
    const x = wait => 77 + (wait / points[points.length - 1].wait_seconds) * 635;
    const y = val => 246 - (val - low) / (high - low) * 205;
    function element(name, attributes, text) {
      const e = document.createElementNS(NS, name);
      Object.entries(attributes).forEach(([k, v]) => e.setAttribute(k, String(v)));
      if (text !== undefined) e.textContent = text;
      svg.append(e); return e;
    }
    for (let i = 0; i <= 4; i++) {
      const value = low + (high - low) * i / 4, position = y(value);
      element('line', { x1: 65, x2: 726, y1: position, y2: position, class: 'gridline' });
      element('text', { x: 53, y: position + 4, 'text-anchor': 'end' }, fmt(value, 1));
    }
    const line = property => points.map((s, i) => `${i ? 'L' : 'M'}${x(s.wait_seconds)},${y(s[property])}`).join(' ');
    element('path', { d: `M65,${y(current.target_actual)} L726,${y(current.target_actual)}`, class: 'truth-path' });
    element('path', { d: line('baseline_prediction'), class: 'baseline-path' });
    element('path', { d: line('prediction'), class: 'arrival-path' });
    const selected = points[active];
    element('line', { x1: x(selected.wait_seconds), x2: x(selected.wait_seconds), y1: 31, y2: 246, class: 'active-line' });
    points.forEach((s, i) => {
      element('circle', { cx: x(s.wait_seconds), cy: y(s.prediction), r: i === active ? 7 : 4, class: i === active ? 'point selected' : 'point' });
      element('text', { x: x(s.wait_seconds), y: 273, 'text-anchor': 'middle' }, s.wait_seconds === 0 ? '즉시' : `${s.wait_seconds / 60}분 대기`);
      if (i === current.selected_step) element('text', { x: x(s.wait_seconds), y: 295, 'text-anchor': 'middle', class: 'callout' }, '정책의 확정 시점');
    });
    svg.setAttribute('aria-label', `동일한 ${stamp(current.target_time)} 예측. 선택 시점 ${selected.wait_seconds / 60}분, 예측 ${fmt(selected.prediction)}, 사후 정답 ${fmt(current.target_actual)}. 사후 정답은 정책 입력이 아닙니다.`);
  }
  function renderStep(index) {
    if (!current) return;
    const step = current.steps[index], chosen = current.steps[current.selected_step], final = index === current.steps.length - 1;
    put('forecast', fmt(step.prediction)); put('chosenWait', fmt(chosen.wait_seconds / 60, 0));
    put('availability', fmt(step.window_observed_fraction * 100, 1)); put('compute', fmt(step.compute_ms, 2));
    put('stepLabel', step.wait_seconds ? `${step.wait_seconds / 60}분` : '즉시');
    const waiting = step.action === 'WAIT';
    const postCommit = index > current.selected_step;
    put('actionTitle', postCommit ? '확정 이후의 비교 시점' : waiting ? '새 관측을 기다립니다' : '현재 예측을 확정합니다');
    put('actionIcon', postCommit ? '↔' : waiting ? '◷' : '✓');
    $('actionIcon').classList.toggle('wait', waiting && !postCommit);
    put('actionText', postCommit
      ? '정책은 이미 예측을 확정했습니다. 이 시점은 사후 비교용이며 실제 정책을 다시 실행한 결과가 아닙니다.'
      : final ? '설정한 대기 마감에 도달했습니다. 더 기다리지 않고 같은 대상 시각의 예측을 확정합니다.'
      : waiting ? '추정한 다음 관측의 이득이 대기 기준보다 큽니다. 다음 판단 시점에 도착 상태를 다시 확인합니다.'
      : '추가 대기의 추정 이득이 기준을 넘지 않았습니다. 미래 관측을 미리 확인하지 않고 결정합니다.');
    put('expectedGain', step.expected_next_error_reduction === null ? '마지막 시점 · 계산 안 함' : fmt(step.expected_next_error_reduction, 4));
    put('requiredGain', step.minimum_gain_to_wait === null ? '마감 도달' : fmt(step.minimum_gain_to_wait, 4));
    put('selectedText', chosen.wait_seconds ? `${chosen.wait_seconds / 60}분 후 확정` : '즉시 확정');
    const body = $('sensors'); body.replaceChildren();
    metadata.columns.forEach((name, i) => {
      const row = document.createElement('tr'); cell(row, name); cell(row, fmt(step.latest_values[i], 3));
      const badge = document.createElement('span'), known = step.latest_values[i] !== null;
      badge.className = `sensor-status${!known ? ' missing' : step.latest_observed[i] ? '' : ' stale'}`;
      badge.textContent = !known ? '사용할 관측 없음' : step.latest_observed[i] ? '도착 완료' : '이전 관측 사용';
      const td = document.createElement('td'); td.append(badge); row.append(td);
      cell(row, step.latest_source_times[i] === null ? '—' : stamp(step.latest_source_times[i]));
      cell(row, step.latest_age_seconds[i] === null ? '—' : `${fmt(step.latest_age_seconds[i] / 60, 0)}분`);
      body.append(row);
    });
    drawChart(index);
  }
  async function loadCase() {
    if (!metadata) return;
    const id = Number($('caseId').value);
    if (!Number.isInteger(id) || id < 0 || id >= metadata.cases) {
      status(`사례 번호는 0부터 ${metadata.cases - 1}까지의 정수로 입력하세요.`, true); return;
    }
    if (controller) controller.abort();
    controller = new AbortController(); const request = ++requestNumber;
    $('runButton').disabled = true; status('저장된 모델로 예측과 대기 판단을 계산하고 있습니다.');
    const scenario = $('scenario').value;
    try {
      const result = await getJSON(`/api/replay?case_id=${id}&scenario=${encodeURIComponent(scenario)}`, controller.signal);
      if (request !== requestNumber) return;
      current = result;
      put('origin', stamp(result.origin_time)); put('targetTime', stamp(result.target_time));
      $('stepRange').max = result.steps.length - 1;
      $('stepRange').value = result.selected_step;
      renderStep(result.selected_step); evidence(scenario); status('');
      $('prevCase').disabled = id === 0; $('nextCase').disabled = id === metadata.cases - 1;
    } catch (error) {
      if (error.name !== 'AbortError' && request === requestNumber) status(`사례 실행 실패: ${error.message}`, true);
    } finally {
      if (request === requestNumber) $('runButton').disabled = false;
    }
  }
  $('runButton').addEventListener('click', loadCase);
  $('scenario').addEventListener('change', loadCase);
  $('caseId').addEventListener('keydown', e => { if (e.key === 'Enter') loadCase(); });
  $('stepRange').addEventListener('input', e => renderStep(Number(e.target.value)));
  for (const [id, delta] of [['prevCase', -1], ['nextCase', 1]]) {
    $(id).addEventListener('click', () => {
      if (!metadata) return;
      $('caseId').value = Math.max(0, Math.min(metadata.cases - 1, Number($('caseId').value) + delta)); loadCase();
    });
  }
  (async () => {
    try {
      metadata = await getJSON('/api/metadata');
      put('modelId', metadata.run_id); put('runShort', metadata.run_id.slice(0, 10));
      put('horizon', metadata.config.horizon * metadata.source.grid_seconds / 3600);
      put('sensorCount', `${metadata.columns.length} channels`); $('caseId').max = metadata.cases - 1;
      const synthetic = metadata.source_kind === 'synthetic';
      put('sourceBadge', synthetic ? '합성 데이터 검증' : metadata.source_kind === 'ett' ? 'ETT 실측값' : '사용자 제공 데이터');
      put('sourceText', synthetic
        ? '측정값과 수집 지연 모두 생성한 데이터입니다. 실측 ETT 성능이나 실제 설비 운영 결과가 아닙니다.'
        : '수집 지연은 합성 조건입니다. 측정 시각은 데이터셋 기준이며 원본의 시간대를 별도로 확인해야 합니다.');
      await loadCase();
    } catch (error) { status(`모델을 불러오지 못했습니다. verify 명령으로 체크포인트를 확인하세요. ${error.message}`, true); }
  })();
})();
