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

  function selection() {
    return {case_id: $('caseId').value.trim(), scenario: $('scenario').value,
      wait_seconds: $('waitSelect').value};
  }

  function selectionKey(value = selection()) {
    return JSON.stringify(value);
  }

  function syncActionButtons() {
    const blocked = busy || !state || stateKey !== selectionKey();
    document.querySelectorAll('.sensor-acquire, .table-action').forEach(button => {
      button.disabled = blocked || button.dataset.eligible !== 'true';
    });
    $('acquireRecommended').disabled = blocked || state.recommended_action !== 'ACQUIRE'
      || !state.recommended_sensor;
  }

  function beginRequest() {
    if (controller) controller.abort();
    controller = new AbortController();
    const value = selection();
    const op = {seq: ++requestSeq, value, key: selectionKey(value), signal: controller.signal};
    busy = true;
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
    $('runButton').disabled = false;
    syncActionButtons();
  }

  function renderState() {
    renderDecision();
    renderSensorMap();
    renderCounterfactual();
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

  function actionCopy() {
    if (!state) return {title: '계산 중', reason: ''};
    if (state.recommended_action === 'ACQUIRE') {
      const candidate = candidateBySensor(state.recommended_sensor);
      return {
        title: `${state.recommended_sensor} 센서를 추가 취득`,
        reason: `예상 오차 감소 ${signed(candidate?.predicted_gain)} · 상대 비용 ${fmt(candidate?.cost_proxy, 2)}. 기다림과 확정의 순효용보다 높은 후보입니다.`
      };
    }
    if (state.recommended_action === 'WAIT') {
      return {
        title: `${Math.round((state.next_wait_seconds || 0) / 60)}분 판단 시점까지 대기`,
        reason: `현재 snapshot에서 passive arrival의 예상 이득 ${signed(state.wait_gain_predicted)}이 acquisition 후보의 순효용보다 큽니다.`
      };
    }
    return {
      title: '현재 예측을 확정',
      reason: '추가 sensor acquisition과 passive wait의 추정 순효용이 현재 확정 기준을 넘지 않았습니다.'
    };
  }

  function renderDecision() {
    $('forecast').textContent = fmt(state.prediction, 2);
    $('disagreement').textContent = fmt(state.disagreement_proxy.value, 3);
    $('availableSensors').textContent = state.available_origin_sensors;
    $('availableTotal').textContent = `/ ${state.total_sensors}`;
    $('actionBadge').textContent = state.recommended_action;
    $('actionBadge').className = `action-badge ${state.recommended_action.toLowerCase()}`;

    const copy = actionCopy();
    $('recommendationTitle').textContent = copy.title;
    $('recommendationReason').textContent = copy.reason;
    const button = $('acquireRecommended');
    const canAcquire = state.recommended_action === 'ACQUIRE' && !!state.recommended_sensor;
    button.disabled = !canAcquire;
    button.dataset.sensor = canAcquire ? state.recommended_sensor : '';
    button.innerHTML = canAcquire
      ? `${state.recommended_sensor} 취득 실행 <b>↗</b>`
      : '추천 센서 취득 <b>↗</b>';

    const audit = $('auditText');
    audit.replaceChildren();
    const top = state.candidates.filter(row => row.eligible).slice(0, 3);
    const blocks = [
      ['Recommendation', copy.title],
      ['Wait value', `${signed(state.wait_gain_predicted)} native error`],
      ['Model disagreement', `${fmt(state.disagreement_proxy.value, 3)} · uncertainty proxy only`],
      ['Already acquired', acquired.length ? acquired.join(' → ') : 'none']
    ];
    blocks.forEach(([label, value]) => {
      const div = document.createElement('div');
      const span = document.createElement('span'); span.textContent = label;
      const strong = document.createElement('strong'); strong.textContent = value;
      div.append(span, strong); audit.append(div);
    });
    if (top.length) {
      const p = document.createElement('p');
      p.className = 'audit-ranked';
      p.textContent = 'Top candidates · ' + top.map((row, i) =>
        `#${i + 1} ${row.sensor} (${signed(row.predicted_gain)})`).join('  ·  ');
      audit.append(p);
    }
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
      rank.textContent = row.eligible ? `#${index + 1}` : got ? 'PULL' : 'LIVE';
      const title = document.createElement('strong'); title.textContent = row.sensor;
      const badge = document.createElement('small');
      badge.textContent = got ? 'actively acquired' : observed ? 'already available' : 'candidate';
      head.append(rank, title, badge);

      const metrics = document.createElement('div');
      metrics.className = 'sensor-card-metrics';
      metrics.innerHTML = `
        <div><span>predicted Δerror</span><b>${signed(row.predicted_gain)}</b></div>
        <div><span>cost proxy</span><b>${fmt(row.cost_proxy, 2)}</b></div>
        <div><span>net utility</span><b>${signed(row.utility)}</b></div>`;

      const button = document.createElement('button');
      button.className = 'sensor-acquire';
      button.textContent = row.eligible ? '이 센서 취득' : got ? '취득 완료' : '이미 도착';
      button.dataset.eligible = String(row.eligible);
      button.disabled = !row.eligible;
      button.addEventListener('click', () => acquireSensor(row.sensor));
      card.append(head, metrics, button);
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
      td(tr, row.already_acquired ? 'acquired' : row.passively_observed_origin ? 'available' : 'candidate');
      td(tr, signed(row.predicted_gain));
      td(tr, signed(row.realized_gain_retrospective), 'audit-value');
      td(tr, fmt(row.cost_proxy, 2));
      td(tr, signed(row.utility));
      const action = document.createElement('td');
      const button = document.createElement('button');
      button.className = 'table-action';
      button.textContent = row.eligible ? 'ACQUIRE' : '—';
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
    const label = document.createElement('small'); label.textContent = event.kind === 'ACTIVE_ACQUISITION' ? 'ACTIVE PULL' : 'PASSIVE ARRIVAL';
    const title = document.createElement('strong'); title.textContent = event.label;
    const value = document.createElement('b'); value.textContent = fmt(event.prediction, 2);
    content.append(label, title, value);
    item.append(marker, content);
    return item;
  }

  async function renderRevisionTimeline(op) {
    const query = new URLSearchParams({case_id: op.value.case_id, scenario: op.value.scenario});
    const payload = await requestJSON(`/api/revision-timeline?${query}`, {signal: op.signal});
    if (!isCurrent(op)) return;
    const root = $('revisionTimeline'); root.replaceChildren();
    payload.events.forEach(event => root.append(timelineNode(event)));
    sessionEvents.forEach(event => root.append(timelineNode(event)));
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
    el('text', {x: 310, y: 302, 'text-anchor': 'middle', class: 'axis-label'}, 'mean relative acquisition cost');
    el('text', {x: 12, y: 160, transform: 'rotate(-90 12 160)', 'text-anchor': 'middle', class: 'axis-label'}, 'MAE');
    $('paretoSummary').textContent =
      `비용 가중치 ${points.length}개를 같은 test set에 적용 · cost는 train-derived relative proxy`;
  }

  async function loadPareto() {
    const payload = await requestJSON('/api/pareto');
    renderPareto(payload.points || []);
  }

  async function loadState({reset = false} = {}) {
    if (!metadata) return;
    if (reset) { acquired = []; sessionEvents = []; }
    state = null;
    stateKey = null;
    const op = beginRequest();
    status('현재 정보로 센서 취득·대기·확정의 가치를 다시 계산하고 있습니다.');
    try {
      const caseId = Number(op.value.case_id);
      if (!op.value.case_id || !Number.isInteger(caseId) || caseId < 0 || caseId >= metadata.cases) {
        throw new Error(`사례 번호는 0부터 ${metadata.cases - 1}까지 정수로 입력해 주세요.`);
      }
      const params = new URLSearchParams({...op.value, acquired: acquired.join(',')});
      const payload = await requestJSON(`/api/acquisition?${params}`, {signal: op.signal});
      if (!isCurrent(op)) return;
      state = payload;
      stateKey = op.key;
      renderState();
      await renderRevisionTimeline(op);
      if (isCurrent(op)) status('');
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrent(op)) status(`계산 실패: ${error.message}`, true);
    } finally {
      finishRequest(op);
    }
  }

  async function acquireSensor(sensor) {
    if (busy || !state || stateKey !== selectionKey() || !candidateBySensor(sensor)?.eligible) return;
    const before = state.prediction;
    const previousAcquired = [...acquired];
    const op = beginRequest();
    status(`${sensor}의 원래 예측 기준 시각 값을 취득하고 있습니다.`);
    try {
      const payload = await requestJSON('/api/acquire', {
        method: 'POST', signal: op.signal,
        body: JSON.stringify({
          case_id: Number(op.value.case_id),
          scenario: op.value.scenario,
          wait_seconds: Number(op.value.wait_seconds),
          acquired: previousAcquired, sensor
        })
      });
      if (!isCurrent(op)) return;
      acquired = payload.acquired;
      state = payload;
      stateKey = op.key;
      sessionEvents.push({
        kind: 'ACTIVE_ACQUISITION',
        label: `${sensor} 취득 · ${fmt(before, 2)} → ${fmt(payload.prediction, 2)}`,
        prediction: payload.prediction
      });
      renderState();
      await renderRevisionTimeline(op);
      if (isCurrent(op)) status('');
    } catch (error) {
      if (error.name !== 'AbortError' && isCurrent(op)) status(`센서 취득 실패: ${error.message}`, true);
    } finally {
      finishRequest(op);
    }
  }

  $('caseId').addEventListener('input', () => {
    if (controller) controller.abort();
    requestSeq++;
    busy = false;
    stateKey = null;
    $('runButton').disabled = false;
    syncActionButtons();
    status('사례 번호가 변경됐습니다. Enter 또는 사례 다시 계산을 눌러 주세요.');
  });

  $('acquireRecommended').addEventListener('click', () => acquireSensor($('acquireRecommended').dataset.sensor));
  $('runButton').addEventListener('click', () => loadState({reset: true}));
  $('resetAcquisition').addEventListener('click', () => loadState({reset: true}));
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

  (async () => {
    try {
      metadata = await requestJSON('/api/metadata');
      $('caseId').max = metadata.cases - 1;
      $('modelName').textContent = metadata.serving_forecaster;
      $('modelId').textContent = metadata.run_id;
      $('runFooter').textContent = `run ${metadata.run_id.slice(0, 10)}`;
      const synthetic = metadata.source_kind === 'synthetic';
      $('currentSource').textContent = synthetic ? '합성 데이터 결과'
        : metadata.source_kind === 'ett' ? 'ETTh1 측정값' : '제공된 측정값';
      $('deploymentLabel').textContent = metadata.cloud_deployed ? '클라우드 환경 설정' : '로컬/검증 환경';
      $('serviceStatus').textContent = '모델 연결 확인';
      $('sourceBadge').textContent = synthetic ? '합성 데모 · model-backed' : metadata.source_kind === 'ett' ? 'ETT measurements' : 'provided data';
      $('sourceText').textContent = synthetic
        ? '현재 공개 데모의 measurement와 arrival은 합성입니다. ETTh1 실측 측정값 검증은 CI evidence로 별도 기록합니다.'
        : '측정값은 데이터셋 원본이며 arrival timestamp는 synthetic condition입니다.';
      await Promise.all([loadPareto(), loadState({reset: true})]);
    } catch (error) {
      status(`모델을 불러오지 못했습니다: ${error.message}`, true);
    }
  })();
})();
