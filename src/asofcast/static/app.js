'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const fmt = (value, digits = 2) => Number.isFinite(value)
    ? Number(value).toLocaleString('ko-KR', {minimumFractionDigits: digits, maximumFractionDigits: digits})
    : '—';
  const signed = (value, digits = 3) => Number.isFinite(value)
    ? `${value >= 0 ? '+' : ''}${fmt(value, digits)}` : '—';
  const stamp = seconds => new Date(seconds * 1000).toISOString().slice(0, 16).replace('T', ' ');

  let metadata = null;
  let state = null;
  let acquired = [];
  let sessionEvents = [];
  let controller = null;
  let requestSeq = 0;
  let busy = false;
  let stateKey = null;
  let baseline = null;
  let committed = false;
  let lastChange = null;
  let comparison = null;
  let compareController = null;
  let compareSeq = 0;
  let compareBusy = false;

  function selection() {
    return {case_id: $('caseId').value.trim(), scenario: $('scenario').value,
      wait_seconds: $('waitSelect').value};
  }

  function selectionKey(value = selection()) {
    return JSON.stringify(value);
  }

  function nextWait() {
    const now = Number($('waitSelect').value);
    return metadata?.config.waits_seconds.find(wait => wait > now);
  }

  function syncActionButtons() {
    const blocked = busy || !state || stateKey !== selectionKey() || committed;
    document.querySelectorAll('.sensor-acquire, .table-action').forEach(button => {
      button.disabled = blocked || button.dataset.eligible !== 'true';
    });
    $('followRecommendation').disabled = blocked;
    $('commitPrediction').disabled = blocked;
    $('waitNext').disabled = blocked || nextWait() === undefined;
    $('waitNext').textContent = nextWait() === undefined ? '마지막 판단 시점'
      : `${duration(nextWait())} 시점 보기`;
    $('compareChoices').disabled = busy || compareBusy || !state || stateKey !== selectionKey();
    $('previewChoices').disabled = $('compareChoices').disabled;
    $('compareSensor').disabled = busy || compareBusy || !state || !state.candidates.some(row => row.eligible);
  }

  function duration(seconds) {
    return seconds % 3600 === 0 ? `${seconds / 3600}시간` : `${fmt(seconds / 60, 0)}분`;
  }

  function beginRequest() {
    invalidateComparison();
    if (controller) controller.abort();
    controller = new AbortController();
    const value = selection();
    const op = {seq: ++requestSeq, value, key: selectionKey(value), signal: controller.signal};
    busy = true;
    $('resultPanel').classList.add('pending');
    $('resultPanel').setAttribute('aria-busy', 'true');
    $('runButton').disabled = true;
    syncActionButtons();
    return op;
  }

  function isCurrent(op) {
    return op.seq === requestSeq && op.key === selectionKey();
  }

  function finishRequest(op) {
    if (!isCurrent(op)) return;
    busy = false;
    $('resultPanel').classList.remove('pending');
    $('resultPanel').setAttribute('aria-busy', 'false');
    $('runButton').disabled = false;
    syncActionButtons();
  }

  function renderState() {
    renderDecision();
    renderSensorMap();
    renderCounterfactual();
    renderContext();
    renderResult();
    renderSessionTimeline();
    syncActionButtons();
  }

  function status(message = '', error = false) {
    $('status').textContent = message;
    $('status').classList.toggle('error', error);
  }

  async function requestJSON(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {Accept: 'application/json', 'Content-Type': 'application/json', ...(options.headers || {})}
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(`${response.status}: ${payload.detail || '요청을 처리하지 못했습니다.'}`);
    }
    return response.json();
  }

  function td(row, value, className = '') {
    const cell = document.createElement('td');
    cell.textContent = value;
    if (className) cell.className = className;
    row.append(cell);
    return cell;
  }

  function candidateBySensor(sensor) {
    return state?.candidates?.find(row => row.sensor === sensor);
  }

  function sensorName(sensor) {
    if (!sensor) return '추가 센서 없음';
    if (sensor === metadata.config.target) return `예측 대상 (${sensor})`;
    const index = state.columns.filter(name => name !== metadata.config.target).indexOf(sensor);
    return index < 0 ? sensor : `보조 센서 ${index + 1} (${sensor})`;
  }

  function actionCopy() {
    if (!state) return {title: '계산 중', reason: '', button: '추천 확인 중'};
    if (state.recommended_action === 'ACQUIRE') {
      return {
        title: `${sensorName(state.recommended_sensor)}를 더 확인해 보세요`,
        reason: '현재 들어온 정보로 비교한 결과, 이 센서를 추가로 읽는 편이 기다리거나 바로 확정하는 것보다 유리할 것으로 계산됐습니다.',
        button: `추천대로 ${state.recommended_sensor} 읽기`,
        help: '실제 장비를 조작하지 않습니다. 기준 시각의 측정값을 공개하고 예측을 다시 계산합니다.'
      };
    }
    if (state.recommended_action === 'WAIT') {
      return {
        title: `${duration(state.next_wait_seconds)} 시점의 정보를 더 기다려보세요`,
        reason: '지금 확정하거나 센서를 더 읽는 것보다, 다음 판단 시점까지 도착하는 정보를 기다리는 편으로 계산됐습니다.',
        button: `추천대로 ${duration(state.next_wait_seconds)} 시점 재생`,
        help: '실제 시간을 기다리지 않습니다. 버튼을 누르면 해당 시점까지 도착한 데이터로 바로 다시 계산합니다.'
      };
    }
    return {
      title: '현재 예측으로 확정해도 되겠습니다',
      reason: '추가로 읽거나 기다리는 데 드는 비용에 비해, 예측이 나아질 것으로 기대되는 이득이 충분하지 않습니다.',
      button: '추천대로 이 예측 확정',
      help: '실제 운영 결정이 아닌 화면 내 확정입니다. 서버 저장이나 장비 제어는 하지 않습니다.'
    };
  }

  function renderDecision() {
    $('forecast').textContent = fmt(state.prediction, 2);
    $('disagreement').textContent = fmt(state.disagreement_proxy.value, 3);
    $('availableSensors').textContent = state.available_origin_sensors;
    $('availableTotal').textContent = `/ ${state.total_sensors}`;
    const labels = {ACQUIRE: '추가 확인', WAIT: '대기', COMMIT: '확정'};
    $('actionBadge').textContent = committed ? '확정됨' : labels[state.recommended_action];
    $('actionBadge').className = `action-badge ${state.recommended_action.toLowerCase()}`;
    const copy = actionCopy();
    $('recommendationTitle').textContent = committed ? '이 화면에서 예측을 확정했습니다' : copy.title;
    $('recommendationReason').textContent = committed
      ? '확정한 값은 이 체험에만 남습니다. 다른 선택을 비교하려면 이 사례를 처음부터 시작하세요.' : copy.reason;
    $('followRecommendation').textContent = committed ? '확정 완료' : copy.button;
    $('actionHelp').textContent = copy.help;
    const audit = $('auditText');
    audit.replaceChildren();
    const top = state.candidates.filter(row => row.eligible).slice(0, 3);
    const blocks = [
      ['현재 모델의 추천', copy.title],
      ['대기의 예상 오차 감소', `${signed(state.wait_gain_predicted)} · 원본 값 단위`],
      ['모델 간 예측 차이', `${fmt(state.disagreement_proxy.value, 3)} · 확률이 아닙니다`],
      ['이번 체험에서 추가로 읽은 센서', acquired.length ? acquired.join(' → ') : '없음']
    ];
    blocks.forEach(([label, value]) => {
      const div = document.createElement('div');
      const span = document.createElement('span'); span.textContent = label;
      const strong = document.createElement('strong'); strong.textContent = value;
      div.append(span, strong); audit.append(div);
    });
    if (top.length) {
      const text = document.createElement('p');
      text.className = 'audit-ranked';
      text.textContent = '추가 확인 후보 · ' + top.map((row, i) =>
        `${i + 1}순위 ${row.sensor} (예상 오차 감소 ${signed(row.predicted_gain)})`).join(' / ');
      audit.append(text);
    }
  }

  function renderContext() {
    const target = metadata.config.target;
    $('forecastSubject').textContent = `예측 대상 센서 (${target})`;
    $('horizonLabel').textContent = `${duration(state.target_time - state.origin_time)} 뒤`;
    for (const [id, key] of [['originTime', 'origin_time'], ['decisionTime', 'decision_time'], ['targetTime', 'target_time']]) {
      $(id).textContent = stamp(state[key]);
      $(id).dataset.timestamp = String(state[key]);
    }
    const missing = state.total_sensors - state.available_origin_sensors;
    $('missingSensors').textContent = String(missing);
    $('caseSummary').textContent = missing
      ? `${state.total_sensors}개 중 ${state.available_origin_sensors}개 센서만 기준 시각의 측정값을 확보했습니다.`
      : `${state.total_sensors}개 센서의 기준 시각 값을 모두 확보했습니다.`;
    $('forecastUnit').textContent = `${target} · ${metadata.source_kind === 'synthetic' ? '합성 값, 물리 단위 미지정' : '데이터셋 원본 값 단위'}`;
    $('arrivalStrip').replaceChildren(...state.columns.map(sensor => {
      const row = candidateBySensor(sensor);
      const chip = document.createElement('span');
      const received = row.passively_observed_origin || row.already_acquired;
      chip.className = `arrival-chip ${received ? 'received' : 'missing'}`;
      const label = row.already_acquired ? '직접 읽음' : received ? '도착' : '미도착';
      chip.textContent = `${received ? '●' : '○'} ${sensorName(sensor).replace('예측 ', '').replace(' 센서 ', ' ')}`;
      chip.title = `${sensorName(sensor)} · ${label}`;
      chip.setAttribute('aria-label', chip.title);
      return chip;
    }));
    const candidates = state.candidates.filter(row => row.eligible);
    const selectedSensor = $('compareSensor').value;
    $('compareSensor').replaceChildren(...(candidates.length ? candidates : [{sensor: '', label: '모두 도착함'}]).map(row => {
      const option = document.createElement('option'); option.value = row.sensor;
      option.textContent = row.label || sensorName(row.sensor); return option;
    }));
    if (candidates.some(row => row.sensor === selectedSensor)) $('compareSensor').value = selectedSensor;
  }

  function renderResult() {
    const before = lastChange ? lastChange.before : baseline.prediction;
    $('beforeForecast').textContent = fmt(before, 2);
    $('afterForecast').textContent = lastChange ? fmt(state.prediction, 2) : '—';
    $('sessionStatus').textContent = committed ? '화면에서 확정'
      : lastChange ? lastChange.actor : '선택 전';
    $('resultTitle').textContent = committed ? '이 예측으로 확정했습니다'
      : lastChange ? '선택 후 이렇게 달라졌습니다' : '선택하면 무엇이 달라질까요?';
    $('forecastDelta').textContent = lastChange
      ? `예측값 변화 ${signed(state.prediction - before, 2)} · 같은 목표 시각`
      : '아직 선택하지 않았습니다.';
    $('resultGuide').hidden = !!lastChange;
    $('actionResources').hidden = !lastChange;
    const extraWait = lastChange?.wait_seconds || 0;
    const extraCost = lastChange?.cost_proxy || 0;
    $('actionWait').textContent = extraWait ? duration(extraWait) : '없음';
    $('actionWait').dataset.value = String(extraWait);
    $('actionCost').textContent = `${fmt(extraCost, 2)} · 상대값`;
    $('actionCost').dataset.value = String(extraCost);
    $('resultExplanation').textContent = !lastChange
      ? '위의 ‘세 선택의 결과 바로 비교’로 먼저 살펴보거나, 모델 추천을 실행해 보세요.'
      : committed
        ? `${lastChange.actor}으로 현재 예측을 이 화면에서 확정했습니다. 서버에 저장하거나 실제 장비에 명령을 보내지 않습니다.`
        : lastChange.kind === 'WAIT'
          ? `${lastChange.actor}으로 ${duration(Number($('waitSelect').value))} 시점까지의 도착 정보를 재생했습니다. 예측 목표는 그대로 두고 사용할 수 있는 입력만 갱신했습니다.`
          : `${lastChange.actor}으로 ${sensorName(lastChange.sensor)}의 기준 시각 값을 읽었습니다. 추가 정보로 예측을 다시 계산했으며, 목표 시각은 바뀌지 않았습니다.`;
    const actual = state.target_actual_retrospective;
    const audit = $('retrospectiveResult');
    if (!lastChange) {
      audit.textContent = '먼저 한 번 선택해 보세요.';
    } else if (Number.isFinite(actual)) {
      const oldError = Math.abs(before - actual), newError = Math.abs(state.prediction - actual);
      const gain = oldError - newError;
      audit.textContent = `사후 정답 ${fmt(actual, 2)} · 절대 오차 ${fmt(oldError, 3)} → ${fmt(newError, 3)}. `
        + (Math.abs(gain) < 1e-8 ? '이번 선택으로 오차는 변하지 않았습니다.'
          : gain > 0 ? `이 사례에서는 오차가 ${fmt(gain, 3)} 줄었습니다.`
          : `이 사례에서는 오차가 ${fmt(-gain, 3)} 늘었습니다.`)
        + ' 한 사례의 결과이며 전체 성능 우위를 뜻하지 않습니다.';
    } else {
      audit.textContent = '이 사례의 사후 정답은 제공되지 않았습니다.';
    }
  }

  function renderSessionTimeline() {
    $('revisionTimeline').replaceChildren(...sessionEvents.map(timelineNode));
  }

  function applyState(payload, op, change = null) {
    state = payload;
    stateKey = op.key;
    acquired = [...payload.acquired];
    if (!baseline) {
      baseline = {prediction: payload.prediction, target_time: payload.target_time};
      sessionEvents.push({kind: 'INITIAL', actor: '체험 시작', label: '처음 예측', prediction: payload.prediction});
    }
    if (change) {
      lastChange = change;
      sessionEvents.push({
        ...change,
        label: change.kind === 'WAIT' ? `${duration(Number(op.value.wait_seconds))} 대기 재생`
          : `${change.sensor} 취득 · ${fmt(change.before, 2)} → ${fmt(payload.prediction, 2)}`,
        prediction: payload.prediction
      });
    }
    renderState();
  }

  function clearSession() {
    acquired = []; sessionEvents = []; baseline = null; lastChange = null; committed = false;
    $('resultEvaluation').open = false;
  }

  function clearStaleView() {
    invalidateComparison();
    $('arrivalStrip').replaceChildren();
    for (const id of ['forecast', 'beforeForecast', 'afterForecast', 'forecastDelta', 'availableSensors', 'missingSensors']) $(id).textContent = '—';
    for (const id of ['originTime', 'decisionTime', 'targetTime']) { $(id).textContent = '—'; delete $(id).dataset.timestamp; }
    $('sessionStatus').textContent = '확인 중';
    $('resultExplanation').textContent = '현재 선택한 사례의 계산 결과를 기다리고 있습니다.';
    $('recommendationTitle').textContent = '현재 선택을 계산하고 있습니다';
    $('recommendationReason').textContent = '계산이 끝나면 추천과 선택 버튼이 활성화됩니다.';
    $('retrospectiveResult').textContent = '계산 완료 후 확인할 수 있습니다.';
    $('actionResources').hidden = true;
    $('resultGuide').hidden = false;
    $('revisionTimeline').replaceChildren();
  }

  function renderSensorMap() {
    const grid = $('sensorMapGrid');
    grid.replaceChildren();
    state.candidates.forEach((row, index) => {
      const card = document.createElement('article');
      const observed = row.passively_observed_origin;
      const got = row.already_acquired;
      card.className = `sensor-card ${got ? 'is-acquired' : observed ? 'is-observed' : 'is-candidate'}`;
      if (state.recommended_sensor === row.sensor && state.recommended_action === 'ACQUIRE') {
        card.classList.add('recommended');
      }

      const head = document.createElement('div');
      head.className = 'sensor-card-head';
      const rank = document.createElement('span');
      rank.className = 'sensor-rank';
      rank.textContent = row.eligible ? `${index + 1}순위` : got ? '확인' : '도착';
      const title = document.createElement('strong'); title.textContent = row.sensor;
      const role = document.createElement('p'); role.className = 'sensor-role';
      role.textContent = sensorName(row.sensor);
      const badge = document.createElement('small');
      badge.textContent = got ? '이번 체험에서 읽음' : observed ? '기준 시각 값 도착' : '기준 시각 값 미도착';
      head.append(rank, title, badge);

      const metrics = document.createElement('div');
      metrics.className = 'sensor-card-metrics';
      metrics.innerHTML = `
        <div><span>예상 오차 감소</span><b>${signed(row.predicted_gain)}</b></div>
        <div><span>상대 정보 비용</span><b>${fmt(row.cost_proxy, 2)}</b></div>
        <div><span>정책 점수</span><b>${signed(row.utility)}</b></div>`;

      const button = document.createElement('button');
      button.className = 'sensor-acquire';
      button.textContent = row.eligible ? '직접 이 센서 읽기' : got ? '읽기 완료' : '이미 도착';
      button.dataset.eligible = String(row.eligible);
      button.disabled = !row.eligible;
      button.addEventListener('click', () => acquireSensor(row.sensor));
      card.append(head, role, metrics, button);
      grid.append(card);
    });
  }

  function renderCounterfactual() {
    const body = $('counterfactualRows');
    body.replaceChildren();
    state.candidates.forEach(row => {
      const tr = document.createElement('tr');
      if (state.recommended_sensor === row.sensor && state.recommended_action === 'ACQUIRE') tr.className = 'row-recommended';
      td(tr, row.sensor);
      td(tr, row.already_acquired ? '읽음' : row.passively_observed_origin ? '도착' : '미도착');
      td(tr, signed(row.predicted_gain));
      td(tr, signed(row.realized_gain_retrospective), 'audit-value');
      td(tr, fmt(row.cost_proxy, 2));
      td(tr, signed(row.utility));
      const action = document.createElement('td');
      const button = document.createElement('button');
      button.className = 'table-action';
      button.textContent = row.eligible ? '직접 읽기' : '—';
      button.dataset.eligible = String(row.eligible);
      button.disabled = !row.eligible;
      button.addEventListener('click', () => acquireSensor(row.sensor));
      action.append(button);
      tr.append(action);
      body.append(tr);
    });
  }

  function timelineNode(event) {
    const item = document.createElement('article');
    item.className = `revision-event ${event.kind === 'ACTIVE_ACQUISITION' ? 'active' : ''}`;
    const marker = document.createElement('span'); marker.className = 'revision-marker';
    const content = document.createElement('div');
    const label = document.createElement('small'); label.textContent = event.actor || '대기만 한 비교 · 미실행';
    const title = document.createElement('strong'); title.textContent = event.label;
    const value = document.createElement('b'); value.textContent = fmt(event.prediction, 2);
    content.append(label, title, value);
    item.append(marker, content);
    return item;
  }

  async function renderRevisionTimeline(op) {
    const query = new URLSearchParams({case_id: op.value.case_id, scenario: op.value.scenario});
    try {
      const payload = await requestJSON(`/api/revision-timeline?${query}`, {signal: op.signal});
      if (!isCurrent(op)) return;
      const root = $('comparisonTimeline'); root.replaceChildren();
      payload.events.filter(event => event.kind === 'PASSIVE').forEach(event => root.append(timelineNode(event)));
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrent(op)) {
        $('comparisonTimeline').textContent = '별도 비교를 불러오지 못했습니다. 현재 예측과 선택 결과는 유지됩니다. 사례 다시 계산으로 재시도할 수 있습니다.';
      }
    }
  }

  function renderPareto(points) {
    const svg = $('paretoChart');
    const NS = 'http://www.w3.org/2000/svg';
    svg.replaceChildren();
    if (!points.length) return;
    const xs = points.map(p => Number(p.mean_cost_proxy));
    const ys = points.map(p => Number(p.mae));
    let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
    if (Math.abs(xmax - xmin) < 1e-9) { xmin -= .05; xmax += .05; }
    if (Math.abs(ymax - ymin) < 1e-9) { ymin -= .01; ymax += .01; }
    const x = v => 74 + (v - xmin) / (xmax - xmin) * 475;
    const y = v => 260 - (v - ymin) / (ymax - ymin) * 195;
    function el(name, attrs, text) {
      const node = document.createElementNS(NS, name);
      Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
      if (text !== undefined) node.textContent = text;
      svg.append(node); return node;
    }
    for (let i = 0; i <= 4; i++) {
      const yy = 65 + 195 * i / 4;
      el('line', {x1: 70, x2: 555, y1: yy, y2: yy, class: 'pareto-grid'});
    }
    const sorted = [...points].sort((a, b) => a.mean_cost_proxy - b.mean_cost_proxy);
    el('path', {
      d: sorted.map((p, i) => `${i ? 'L' : 'M'}${x(p.mean_cost_proxy)},${y(p.mae)}`).join(' '),
      class: 'pareto-line'
    });
    sorted.forEach(p => {
      el('circle', {cx: x(p.mean_cost_proxy), cy: y(p.mae), r: 6, class: 'pareto-point'});
      el('text', {x: x(p.mean_cost_proxy), y: y(p.mae) - 12, 'text-anchor': 'middle'}, `λ ${p.cost_weight}`);
    });
    el('text', {x: 310, y: 302, 'text-anchor': 'middle', class: 'axis-label'}, '센서 추가 확인 상대 비용');
    el('text', {x: 12, y: 160, transform: 'rotate(-90 12 160)', 'text-anchor': 'middle', class: 'axis-label'}, 'MAE');
    $('paretoSummary').textContent =
      `비용 가중치 ${points.length}개를 같은 평가 구간에 적용했습니다. 비용은 학습 구간에서 만든 상대값이며, 금액이나 실제 장비 지연이 아닙니다.`;
  }

  async function loadPareto() {
    const payload = await requestJSON('/api/pareto');
    renderPareto(payload.points || []);
  }

  function invalidateComparison() {
    compareController?.abort();
    compareSeq++;
    compareBusy = false;
    comparison = null;
    $('choiceComparison').replaceChildren();
    $('comparisonInsight').textContent = '';
    delete $('comparisonInsight').dataset.complete;
    $('choiceComparison').setAttribute('aria-busy', 'false');
    $('showComparisonTruth').checked = false;
    $('showComparisonTruth').disabled = true;
    $('exportComparison').disabled = true;
    $('compareChoices').textContent = '세 선택 계산하기';
    $('comparisonStatus').textContent = '센서를 고르고 세 선택을 계산해 보세요. 현재 모델 추천은 그대로 유지됩니다.';
  }

  function renderComparison() {
    if (!comparison) return;
    const reveal = $('showComparisonTruth').checked;
    const labels = {COMMIT: '지금 확정', WAIT: '조금 더 기다림', ACQUIRE: '센서 하나 더 읽기'};
    const icons = {COMMIT: '01', WAIT: '02', ACQUIRE: '03'};
    const commitRow = comparison.choices.find(row => row.action === 'COMMIT');
    const actual = comparison.target_actual_retrospective;
    const hasTruth = Number.isFinite(actual);
    const baselineError = Math.abs(commitRow.prediction - actual);
    $('choiceComparison').replaceChildren(...comparison.choices.map(row => {
      const card = document.createElement('article');
      card.className = `choice-card ${row.action.toLowerCase()}`;
      card.dataset.choice = row.action; card.dataset.status = row.status;
      if (row.target_time !== undefined) card.dataset.target = String(row.target_time);
      const tag = document.createElement('span'); tag.className = 'choice-number'; tag.textContent = icons[row.action];
      const title = document.createElement('h3'); title.textContent = labels[row.action];
      const condition = document.createElement('p'); condition.className = 'choice-condition';
      condition.textContent = row.action === 'ACQUIRE' ? `${sensorName(comparison.sensor)} · 기준 시각 값`
        : row.action === 'WAIT' && row.status === 'error' ? '계산할 대기 선택 · 결과 불러오기 실패'
        : row.action === 'WAIT' && row.wait_seconds !== undefined ? `${duration(row.wait_seconds)} 더 기다린 뒤`
        : row.action === 'COMMIT' ? '지금 가진 정보 그대로' : '더 기다릴 시점 없음';
      const forecast = document.createElement('strong'); forecast.className = 'choice-forecast';
      forecast.textContent = row.status === 'ok' ? fmt(row.prediction, 2) : '—';
      if (row.status === 'ok') forecast.dataset.value = String(row.prediction);
      const unit = document.createElement('p'); unit.className = 'choice-unit';
      unit.textContent = metadata.source_kind === 'synthetic' ? '합성 예측값 · 물리 단위 미지정' : '예측값 · 데이터셋 원본 단위';
      const delta = document.createElement('p'); delta.className = 'choice-delta';
      delta.textContent = row.status !== 'ok' ? '' : row.action === 'COMMIT' ? '비교의 기준이 되는 예측'
        : `지금 확정할 때보다 ${signed(row.prediction - commitRow.prediction, 2)}`;
      const resources = document.createElement('dl'); resources.className = 'choice-resources';
      if (row.status === 'ok') {
        for (const [label, className, value, display] of [
          ['추가 대기', 'choice-wait', row.wait_seconds, row.wait_seconds ? duration(row.wait_seconds) : '없음'],
          ['정보 비용', 'choice-cost', row.extra_cost_proxy, `${fmt(row.extra_cost_proxy, 2)} · 상대값`]
        ]) {
          const item = document.createElement('div'), term = document.createElement('dt'), detail = document.createElement('dd');
          term.textContent = label; detail.className = className; detail.dataset.value = String(value);
          detail.textContent = display; item.append(term, detail); resources.append(item);
        }
      }
      const note = document.createElement('p'); note.className = 'choice-note';
      note.textContent = row.status === 'ok'
        ? `${row.available_origin_sensors}/${state.total_sensors}개 센서의 기준 시각 값 사용`
        : row.message;
      const error = document.createElement('div'); error.className = 'choice-error';
      error.hidden = !reveal || row.status !== 'ok' || !hasTruth;
      const verdict = document.createElement('p'); verdict.className = 'choice-verdict';
      verdict.hidden = error.hidden;
      if (row.status === 'ok' && hasTruth) {
        const value = Math.abs(row.prediction - actual);
        const gain = baselineError - value;
        error.dataset.value = String(value);
        error.textContent = `사후 절대 오차 ${fmt(value, 3)}`;
        verdict.dataset.gain = String(gain);
        const direction = Math.abs(gain) < 1e-8 ? 'same' : gain > 0 ? 'better' : 'worse';
        verdict.dataset.direction = direction;
        verdict.textContent = row.action === 'COMMIT' ? '오차 = 예측값과 사후 정답 사이의 거리'
          : direction === 'same' ? '지금 확정할 때와 오차가 같습니다.'
          : `지금 확정보다 오차 ${fmt(Math.abs(gain), 3)} ${direction === 'better' ? '감소' : '증가'}`;
      }
      card.append(tag, title, condition, forecast, unit, delta, resources, note, error, verdict);
      return card;
    }));
    const errors = comparison.choices.filter(row => row.status === 'error').length;
    $('comparisonStatus').textContent = `사례 ${comparison.start.case_id} · 목표 ${stamp(comparison.start.target_time)} 고정. `
      + (errors ? '일부 선택을 계산하지 못했습니다. 세 선택 계산하기로 재시도할 수 있습니다.'
        : '서로 독립된 세 갈래의 비교입니다. 현재 선택과 모델 추천은 바뀌지 않습니다.')
      + (reveal ? ` 사후 정답 ${fmt(comparison.target_actual_retrospective, 2)}.` : '');
    const complete = comparison.choices.every(row => row.status === 'ok');
    const insight = $('comparisonInsight');
    insight.dataset.complete = String(complete);
    insight.textContent = errors
      ? '일부 선택의 계산에 실패했습니다. 계산된 결과만 확인하고 다시 계산해 주세요.'
      : !complete ? '마지막 판단 시점이거나 모든 센서 값이 도착하면 일부 선택은 제공되지 않습니다. 지금 가능한 선택의 결과를 비교하세요.'
      : reveal && hasTruth
        ? '이 사례의 실제 오차를 지금 확정했을 때와 비교했습니다. 오차가 줄어도 추가 대기·정보 비용을 함께 봐야 합니다. 한 사례로 모델 전체의 우위를 판단하지 않습니다.'
        : '세 선택은 사용하는 정보와 기다리는 시간이 다릅니다. 예측값의 변화만으로 더 정확해졌는지는 알 수 없습니다. 아래에서 사후 정답을 켜 확인하세요.';
  }

  async function compareChoices() {
    if (busy || compareBusy || !state || stateKey !== selectionKey()) return;
    invalidateComparison();
    const seq = compareSeq;
    compareController = new AbortController();
    const signal = compareController.signal;
    const startState = state;
    const start = {...selection(), acquired: [...acquired]};
    const sensor = $('compareSensor').value;
    const wait = nextWait();
    compareBusy = true;
    syncActionButtons();
    $('compareChoices').textContent = '같은 출발점으로 계산 중…';
    $('choiceComparison').setAttribute('aria-busy', 'true');
    const record = (action, value, extraCost = 0) => {
      if (value.target_time !== startState.target_time || value.origin_time !== startState.origin_time
          || value.case_id !== startState.case_id || value.run_id !== startState.run_id) {
        throw new Error('비교 조건이 달라 결과를 표시하지 않았습니다.');
      }
      return {action, status: 'ok', prediction: value.prediction, target_time: value.target_time,
        available_origin_sensors: value.available_origin_sensors,
        wait_seconds: value.decision_time - startState.decision_time, extra_cost_proxy: extraCost};
    };
    const branch = async (action, task, unavailable, extraCost = 0) => {
      if (!task) return {action, status: 'unavailable', message: unavailable};
      try { return record(action, await task(), extraCost); }
      catch (error) { return {action, status: 'error', message: '계산 실패 · 다시 비교해 주세요.'}; }
    };
    const outcomes = await Promise.all([
      branch('WAIT', wait === undefined ? null : () => requestJSON('/api/acquisition?' + acquisitionQuery({
        case_id: start.case_id, scenario: start.scenario, wait_seconds: wait
      }, start.acquired), {signal}), '마지막 판단 시점입니다. 더 기다리는 선택은 제공하지 않습니다.'),
      branch('ACQUIRE', sensor ? () => requestJSON('/api/acquire', {method: 'POST', signal,
        body: JSON.stringify({case_id: Number(start.case_id), scenario: start.scenario,
          wait_seconds: Number(start.wait_seconds), acquired: start.acquired, sensor})}) : null,
        '모든 센서가 도착했습니다. 추가 취득할 값이 없습니다.', candidateBySensor(sensor)?.cost_proxy || 0)
    ]);
    if (seq !== compareSeq) return;
    comparison = {schema: 'asofcast.choice-comparison.v1', scope: 'single_case_counterfactual_not_executed',
      run_id: startState.run_id, source_kind: startState.source_kind, sensor,
      start: {case_id: Number(start.case_id), scenario: start.scenario, wait_seconds: Number(start.wait_seconds),
        acquired: start.acquired, origin_time: startState.origin_time, target_time: startState.target_time},
      current_recommendation: startState.recommended_action,
      target_actual_retrospective: startState.target_actual_retrospective,
      choices: [record('COMMIT', startState), ...outcomes]};
    compareBusy = false;
    $('choiceComparison').setAttribute('aria-busy', 'false');
    $('compareChoices').textContent = '세 선택 다시 계산';
    $('showComparisonTruth').disabled = false;
    $('exportComparison').disabled = false;
    renderComparison();
    syncActionButtons();
  }

  function exportComparison() {
    if (!comparison) return;
    const data = {...comparison, choices: comparison.choices.map(row => ({...row}))};
    if (!$('showComparisonTruth').checked) delete data.target_actual_retrospective;
    else data.choices.forEach(row => {
      if (row.status === 'ok') row.absolute_error_retrospective = Math.abs(row.prediction - data.target_actual_retrospective);
    });
    data.limitations = ['Single replay case; not aggregate policy performance.',
      'Arrival delay is synthetic. Information cost is a relative proxy, not money or latency.',
      'Alternative branches were previewed, not executed in the user session.'];
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = `asofcast-case-${data.start.case_id}-comparison.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function acquisitionQuery(values, sensors) {
    const params = new URLSearchParams(values);
    for (const sensor of sensors) params.append('acquired_sensor', sensor);
    return params;
  }

  async function loadState({reset = false, change = null} = {}) {
    if (!metadata) return;
    if (reset) clearSession();
    state = null;
    stateKey = null;
    clearStaleView();
    const op = beginRequest();
    status('현재 들어온 정보로 추가 확인·대기·확정의 가치를 계산하고 있습니다.');
    try {
      const caseId = Number(op.value.case_id);
      if (!op.value.case_id || !Number.isInteger(caseId) || caseId < 0 || caseId >= metadata.cases) {
        throw new Error(`사례 번호는 0부터 ${metadata.cases - 1}까지 정수로 입력해 주세요.`);
      }
      const params = acquisitionQuery(op.value, acquired);
      const payload = await requestJSON(`/api/acquisition?${params}`, {signal: op.signal});
      if (!isCurrent(op)) return;
      applyState(payload, op, change);
      await renderRevisionTimeline(op);
      if (isCurrent(op)) status('');
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrent(op)) {
        status(`계산 실패: ${error.message}`, true);
        if (!state) $('sessionStatus').textContent = '계산 실패 · 다시 계산해 주세요';
      }
    } finally {
      finishRequest(op);
    }
  }

  async function acquireSensor(sensor, actor = '직접 선택') {
    if (busy || committed || !state || stateKey !== selectionKey() || !candidateBySensor(sensor)?.eligible) return;
    const before = state.prediction;
    const cost = candidateBySensor(sensor).cost_proxy;
    const previousAcquired = [...acquired];
    const op = beginRequest();
    status(`${sensor}의 원래 예측 기준 시각 값을 읽고 있습니다.`);
    try {
      const payload = await requestJSON('/api/acquire', {
        method: 'POST', signal: op.signal,
        body: JSON.stringify({case_id: Number(op.value.case_id), scenario: op.value.scenario,
          wait_seconds: Number(op.value.wait_seconds), acquired: previousAcquired, sensor})
      });
      if (!isCurrent(op)) return;
      applyState(payload, op, {kind: 'ACTIVE_ACQUISITION', actor, before, sensor, cost_proxy: cost});
      await renderRevisionTimeline(op);
      if (isCurrent(op)) status('');
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrent(op)) status(`센서 읽기 실패: ${error.message}`, true);
    } finally {
      finishRequest(op);
    }
  }

  function advanceDecision(actor = '직접 선택') {
    if (busy || committed || !state || stateKey !== selectionKey()) return;
    const wait = nextWait();
    if (wait === undefined) return;
    const change = {kind: 'WAIT', actor, before: state.prediction,
      wait_seconds: wait - Number($('waitSelect').value)};
    $('waitSelect').value = String(wait);
    loadState({change});
  }

  function commitPrediction(actor = '직접 선택') {
    if (busy || committed || !state || stateKey !== selectionKey()) return;
    committed = true;
    lastChange = {kind: 'COMMIT', actor, before: state.prediction};
    sessionEvents.push({kind: 'COMMIT', actor, label: '이 화면에서 예측 확정', prediction: state.prediction});
    renderState();
    status('');
  }

  function followRecommendation() {
    if (busy || committed || !state || stateKey !== selectionKey()) return;
    if (state.recommended_action === 'WAIT') advanceDecision('모델 추천');
    else if (state.recommended_action === 'ACQUIRE') acquireSensor(state.recommended_sensor, '모델 추천');
    else if (state.recommended_action === 'COMMIT') commitPrediction('모델 추천');
  }

  $('caseId').addEventListener('input', () => {
    if (controller) controller.abort();
    requestSeq++;
    busy = false;
    stateKey = null;
    clearStaleView();
    $('resultPanel').setAttribute('aria-busy', 'false');
    $('resultPanel').classList.remove('pending');
    $('runButton').disabled = false;
    syncActionButtons();
    status('사례 번호가 변경됐습니다. Enter 또는 사례 다시 계산을 눌러 주세요.');
  });

  $('followRecommendation').addEventListener('click', followRecommendation);
  $('compareChoices').addEventListener('click', compareChoices);
  $('previewChoices').addEventListener('click', () => {
    compareChoices();
    $('comparison').scrollIntoView({
      behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
      block: 'start'
    });
  });
  $('compareSensor').addEventListener('change', () => { invalidateComparison(); syncActionButtons(); });
  $('showComparisonTruth').addEventListener('change', renderComparison);
  $('exportComparison').addEventListener('click', exportComparison);
  $('waitNext').addEventListener('click', () => advanceDecision());
  $('commitPrediction').addEventListener('click', () => commitPrediction());
  $('runButton').addEventListener('click', () => loadState({reset: true}));
  $('resetAcquisition').addEventListener('click', () => {
    if (metadata) $('waitSelect').value = String(metadata.config.waits_seconds[0]);
    loadState({reset: true});
  });
  $('scenario').addEventListener('change', () => loadState({reset: true}));
  $('waitSelect').addEventListener('change', () => loadState({reset: true}));
  $('caseId').addEventListener('keydown', event => { if (event.key === 'Enter') loadState({reset: true}); });
  for (const [id, delta] of [['prevCase', -1], ['nextCase', 1]]) {
    $(id).addEventListener('click', () => {
      if (!metadata) return;
      const current = Number($('caseId').value);
      $('caseId').value = Math.max(0, Math.min(metadata.cases - 1, current + delta));
      loadState({reset: true});
    });
  }

  function openAnchor(hash) {
    if (!hash || !hash.startsWith('#')) return;
    const target = document.getElementById(hash.slice(1));
    if (!target) return;
    if ($('technicalDetails').contains(target)) $('technicalDetails').open = true;
  }
  document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => openAnchor(link.hash)));
  window.addEventListener('hashchange', () => openAnchor(window.location.hash));
  openAnchor(window.location.hash);

  async function loadResearchEvidence() {
    const button = $('reloadResearch');
    button.disabled = true;
    $('researchRows').replaceChildren();
    $('researchVerdict').textContent = '판정 확인 중';
    $('researchStatus').textContent = '보존된 평가 결과를 불러오는 중입니다.';
    try {
      const research = await requestJSON('/static/research-evidence.json');
      if (research.schema !== 'asofcast.demo-research-evidence.v1'
          || research.rows?.length !== research.total_conditions
          || research.rows.some(row => !Number.isFinite(row.improvement_percent)
            || !Number.isFinite(row.lower95) || !Number.isFinite(row.upper95)
            || typeof row.passed !== 'boolean')
          || research.passed_conditions !== research.rows.filter(row => row.passed).length
          || research.gate_passed !== research.rows.every(row => row.passed)
          || research.status !== (research.gate_passed ? 'promoted' : 'rejected')) {
        throw new Error('연구 결과와 판정 불일치');
      }
      $('researchRows').replaceChildren(...research.rows.map(row => {
        const tr = document.createElement('tr');
        const values = [row.condition, `${signed(row.improvement_percent, 2)}%`,
          `[${fmt(row.lower95, 6)}, ${fmt(row.upper95, 6)}]`,
          `${row.improved_seeds} / ${research.training_seeds}`, row.passed ? '수치 기준 통과' : '미달'];
        values.forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
        return tr;
      }));
      $('researchVerdict').textContent = `${research.gate_passed ? '전체 기준 통과' : '전체 기준 미달 · 승격 거절'} — 조건 ${research.passed_conditions} / ${research.total_conditions} 통과`;
      $('researchStatus').textContent = `${research.research_date} 평가 원본과 SHA-256로 연결한 판정입니다. 연구 후보를 현재 체험 모델로 승격하지 않았습니다.`;
    } catch (error) {
      $('researchStatus').textContent = '연구 집계를 읽지 못했습니다. 원문 보고서와 승격 판정 JSON에서 확인하거나 다시 읽어 주세요. 예측 체험은 계속 사용할 수 있습니다.';
    } finally {
      button.disabled = false;
    }
  }
  $('reloadResearch').addEventListener('click', loadResearchEvidence);
  loadResearchEvidence();

  (async () => {
    try {
      metadata = await requestJSON('/api/metadata');
      $('caseId').max = metadata.cases - 1;
      $('modelName').textContent = metadata.serving_forecaster;
      $('servingBackend').textContent = metadata.serving_backend === 'onnxruntime' ? 'ONNX Runtime · CPU' : 'PyTorch · CPU';
      $('runtimeScope').textContent = metadata.serving_backend === 'onnxruntime'
        ? '현재 요청은 검증된 ONNX Runtime 경로에서 계산합니다. 별도 동일 입력 CPU 벤치마크의 지연 수치와 현재 HTTP 응답 시간은 구분합니다.'
        : '현재 요청은 PyTorch · CPU로 계산합니다. ONNX Runtime은 Docker/CI에서 별도로 검증한 선택 경로이며, 동일 입력 벤치마크의 속도 개선을 현재 공개 서비스의 HTTP 응답 시간으로 해석하지 않습니다.';
      $('waitSelect').replaceChildren(...metadata.config.waits_seconds.map(wait => {
        const option = document.createElement('option'); option.value = String(wait);
        option.textContent = wait === 0 ? '기준 시각' : `${duration(wait)} 뒤`;
        return option;
      }));
      $('runFooter').textContent = `모델 실행 기록 ${metadata.run_id.slice(0, 12)}`;
      const synthetic = metadata.source_kind === 'synthetic';
      $('currentSource').textContent = synthetic ? '합성 데이터 결과'
        : metadata.source_kind === 'ett' ? 'ETTh1 측정값' : '제공된 측정값';
      $('deploymentLabel').textContent = metadata.cloud_deployed ? '클라우드 환경 설정' : '로컬/검증 환경';
      $('serviceStatus').textContent = '모델 연결 확인';
      $('sourceBadge').textContent = synthetic ? '합성 데이터 체험' : metadata.source_kind === 'ett' ? 'ETTh1 실측값 재생' : '제공된 데이터 재생';
      $('sourceText').textContent = synthetic
        ? '측정값과 도착 지연을 모두 생성한 가상 사례입니다. 실제 모델이 계산하지만, 실제 설비 관측이나 장비 제어는 아닙니다.'
        : '측정값은 데이터셋 원본, 도착 지연은 합성입니다. 실제 장비를 조작하지 않고 당시 정보를 재생합니다.';
      await Promise.all([loadPareto(), loadState({reset: true})]);
    } catch (error) {
      $('serviceStatus').textContent = '연결 실패';
      status(`모델을 불러오지 못했습니다: ${error.message} 새로고침 후 다시 시도해 주세요.`, true);
    }
  })();
})();
