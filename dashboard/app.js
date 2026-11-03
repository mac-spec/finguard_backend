(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const state = {
    rows: [],
    socket: null,
    reconnectTimer: null,
    pingTimer: null,
    alertEnabled: false,
    soundEnabled: false,
    audioContext: null,
    scanned: 0,
    flagged: 0,
    high: 0,
    recentIds: new Set()
  };

  const setMessage = (message, isError) => {
    const el = $('scan-message');
    el.textContent = message;
    el.classList.toggle('error', Boolean(isError));
  };

  const parseCSV = (text) => {
    const rows = [];
    let row = [];
    let cell = '';
    let quoted = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      if (char === '"') {
        if (quoted && text[i + 1] === '"') { cell += '"'; i += 1; }
        else quoted = !quoted;
      } else if (char === ',' && !quoted) {
        row.push(cell); cell = '';
      } else if ((char === '\n' || char === '\r') && !quoted) {
        if (char === '\r' && text[i + 1] === '\n') i += 1;
        row.push(cell); cell = '';
        if (row.some((part) => part.trim() !== '')) rows.push(row);
        row = [];
      } else {
        cell += char;
      }
    }
    row.push(cell);
    if (row.some((part) => part.trim() !== '')) rows.push(row);
    if (rows.length < 2) throw new Error('The CSV needs a header row and at least one transaction.');
    const headers = rows[0].map((name) => name.trim().toLowerCase().replace(/[\s-]+/g, '_'));
    return rows.slice(1).map((values, index) => {
      const record = {};
      headers.forEach((header, column) => { record[header] = (values[column] || '').trim(); });
      const pick = (keys, fallback) => {
        for (const key of keys) if (record[key] !== undefined && record[key] !== '') return record[key];
        return fallback;
      };
      const rawAmount = String(pick(['amount', 'transaction_amount', 'value'], '0')).replace(/[₹,\s]/g, '');
      const amount = Number(rawAmount);
      const rawDirection = String(pick(['direction', 'type', 'txn_type', 'transaction_type', 'dr_cr'], 'debit')).toLowerCase();
      const direction = /(credit|credited|\bcr\b|incoming|received)/.test(rawDirection) ? 'credit' : 'debit';
      const date = pick(['date', 'timestamp', 'datetime', 'transaction_date', 'time'], new Date().toISOString());
      return {
        id: String(pick(['id', 'txn_id', 'transaction_id', 'transaction_reference', 'ref_no'], 'csv-' + Date.now() + '-' + index)).slice(0, 128),
        date: String(date).slice(0, 64),
        description: String(pick(['description', 'narration', 'remarks', 'merchant_name', 'merchant', 'details'], 'Imported transaction')).slice(0, 500),
        amount: amount,
        direction: direction,
        category: String(pick(['category', 'transaction_category'], 'Other')).slice(0, 80),
        method: String(pick(['method', 'payment_method', 'channel', 'mode'], 'Unknown')).slice(0, 80),
        receiver: String(pick(['receiver', 'recipient', 'counterparty', 'beneficiary', 'merchant'], '')).slice(0, 256),
        status: String(pick(['status', 'transaction_status'], 'Unknown')).slice(0, 80)
      };
    }).filter((item) => Number.isFinite(item.amount) && item.amount > 0);
  };

  const formatAmount = (amount) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(Number(amount || 0));
  const escapeText = (value) => String(value == null ? '' : value);

  const addCell = (row, value, className) => {
    const cell = document.createElement('td');
    if (className) cell.className = className;
    cell.textContent = value;
    row.appendChild(cell);
    return cell;
  };

  const renderResults = (response) => {
    const alerts = Array.isArray(response.alerts) ? response.alerts : [];
    const flaggedIds = new Map(alerts.map((alert) => [String(alert.id), alert]));
    const tbody = $('results-body');
    tbody.replaceChildren();
    const visibleRows = state.rows.slice(0, 12);
    if (!visibleRows.length) {
      const row = document.createElement('tr');
      addCell(row, 'Your latest scan will appear here.', 'empty-cell').colSpan = 3;
      tbody.appendChild(row);
    }
    visibleRows.forEach((transaction) => {
      const row = document.createElement('tr');
      const alert = flaggedIds.get(String(transaction.id));
      const nameCell = document.createElement('td');
      const name = document.createElement('span');
      name.className = 'transaction-name';
      name.textContent = escapeText(transaction.description);
      const meta = document.createElement('span');
      meta.className = 'transaction-meta';
      meta.textContent = escapeText(transaction.id) + ' · ' + escapeText(transaction.method);
      nameCell.append(name, meta);
      row.appendChild(nameCell);
      addCell(row, formatAmount(transaction.amount));
      const reviewCell = document.createElement('td');
      const chip = document.createElement('span');
      chip.className = 'risk-chip ' + (alert ? ((alert.fraud_risk_level === 'HIGH' || alert.trust_classification === 'Manipulated' || alert.explanation_verdict === 'UNTRUSTWORTHY') ? 'high' : 'medium') : 'clear');
      chip.textContent = alert ? ((chip.classList.contains('high')) ? 'HIGH RISK' : 'REVIEW') : 'NO FLAG';
      reviewCell.appendChild(chip);
      row.appendChild(reviewCell);
      tbody.appendChild(row);
    });
    $('batch-count').textContent = state.rows.length + ' records';
    $('metric-scanned').textContent = String(state.scanned);
    $('metric-flagged').textContent = String(state.flagged);
    $('metric-high').textContent = String(state.high);
    $('metric-status').textContent = response.notification_level === 'HIGH' ? 'High risk' : (response.flagged_count ? 'Review needed' : 'Clear');
    $('model-warning').hidden = !response.model_warning;
    $('model-warning').textContent = response.model_warning || '';
  };

  const emptyFeed = () => {
    const empty = document.createElement('div');
    empty.className = 'feed-empty';
    const icon = document.createElement('span');
    icon.className = 'feed-empty-icon';
    icon.textContent = '⌁';
    const title = document.createElement('strong');
    title.textContent = 'No new risk signals';
    const copy = document.createElement('p');
    copy.textContent = 'Activate live alerts, then run a scan. New scan alerts will appear here immediately.';
    empty.append(icon, title, copy);
    return empty;
  };

  const addFeedItem = (alert, timeText) => {
    const feed = $('notification-feed');
    const placeholder = feed.querySelector('.feed-empty');
    if (placeholder) placeholder.remove();
    const item = document.createElement('article');
    const high = alert.risk_level === 'HIGH';
    item.className = 'feed-item' + (high ? ' high' : '');
    const top = document.createElement('div');
    top.className = 'feed-item-top';
    const type = document.createElement('span');
    type.className = 'feed-item-type';
    type.textContent = high ? 'HIGH RISK · URGENT REVIEW' : 'MILD RISK · REVIEW';
    const time = document.createElement('span');
    time.className = 'feed-time';
    time.textContent = timeText || new Date().toLocaleTimeString();
    top.append(type, time);
    const title = document.createElement('div');
    title.className = 'feed-item-title';
    title.textContent = alert.description || 'Transaction flagged for review';
    const detail = document.createElement('div');
    detail.className = 'feed-item-detail';
    detail.textContent = formatAmount(alert.amount) + ' · Ref ' + (alert.transaction_id || 'unidentified');
    item.append(top, title, detail);
    if (Array.isArray(alert.reasons) && alert.reasons.length) {
      const list = document.createElement('ul');
      list.className = 'feed-item-reasons';
      alert.reasons.slice(0, 3).forEach((reason) => {
        const li = document.createElement('li');
        li.textContent = reason;
        list.appendChild(li);
      });
      item.appendChild(list);
    }
    feed.prepend(item);
    while (feed.children.length > 20) feed.lastElementChild.remove();
  };

  const playBeep = async (level) => {
    if (!state.soundEnabled) return;
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    try {
      if (!state.audioContext) state.audioContext = new AudioContextClass();
      if (state.audioContext.state === 'suspended') await state.audioContext.resume();
      const count = level === 'HIGH' ? 3 : 1;
      for (let index = 0; index < count; index += 1) {
        const startAt = state.audioContext.currentTime + index * 0.24;
        const oscillator = state.audioContext.createOscillator();
        const gain = state.audioContext.createGain();
        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(level === 'HIGH' ? 830 : 590, startAt);
        gain.gain.setValueAtTime(0.0001, startAt);
        gain.gain.exponentialRampToValueAtTime(level === 'HIGH' ? 0.16 : 0.11, startAt + 0.012);
        gain.gain.exponentialRampToValueAtTime(0.0001, startAt + 0.14);
        oscillator.connect(gain);
        gain.connect(state.audioContext.destination);
        oscillator.start(startAt);
        oscillator.stop(startAt + 0.15);
      }
    } catch (error) {
      console.warn('Audio alert could not play:', error);
    }
  };

  const showToast = (level, count) => {
    const stack = $('toast-stack');
    const toast = document.createElement('div');
    toast.className = 'toast' + (level === 'HIGH' ? ' high' : '');
    const heading = document.createElement('strong');
    heading.textContent = level === 'HIGH' ? 'HIGH RISK SIGNAL DETECTED' : 'MILD RISK SIGNAL DETECTED';
    const copy = document.createElement('p');
    copy.textContent = count + ' transaction' + (count === 1 ? '' : 's') + ' flagged. Open the live feed to review the reasons.';
    toast.append(heading, copy);
    stack.prepend(toast);
    window.setTimeout(() => toast.remove(), 6500);
  };

  const handleLiveEvent = (event) => {
    if (!event || event.type !== 'scan_alerts' || !Array.isArray(event.alerts)) return;
    const alerts = event.alerts;
    state.scanned += Number(event.transactions_analyzed || 0);
    state.flagged += Number(event.flagged_count || 0);
    state.high += Number(event.high_risk_count || 0);
    $('metric-scanned').textContent = String(state.scanned);
    $('metric-flagged').textContent = String(state.flagged);
    $('metric-high').textContent = String(state.high);
    if (!alerts.length) {
      $('metric-status').textContent = 'Clear';
      return;
    }
    const level = event.notification_level === 'HIGH' || alerts.some((alert) => alert.risk_level === 'HIGH') ? 'HIGH' : 'MEDIUM';
    const timestamp = event.created_at ? new Date(event.created_at).toLocaleTimeString() : new Date().toLocaleTimeString();
    alerts.forEach((alert) => addFeedItem(alert, timestamp));
    $('metric-status').textContent = level === 'HIGH' ? 'High risk' : 'Review needed';
    showToast(level, Number(event.flagged_count || alerts.length));
    playBeep(level);
    if ('Notification' in window && Notification.permission === 'granted') {
      const title = level === 'HIGH' ? 'FinGuard: High risk detected' : 'FinGuard: Mild risk detected';
      new Notification(title, { body: String(event.flagged_count || alerts.length) + ' transaction(s) need review.' });
    }
  };

  const setConnection = (connected, label) => {
    $('connection-state').classList.toggle('connected', connected);
    $('connection-text').textContent = label;
    $('feed-state').textContent = connected ? 'CONNECTED' : 'DISCONNECTED';
    $('metric-status').textContent = connected ? 'Monitoring' : 'Standby';
    document.querySelector('.feed-panel').classList.toggle('connected', connected);
  };

  const connectLiveAlerts = async () => {
    if (!state.alertEnabled) return;
    try {
      const headers = {};
      const key = $('api-key').value.trim();
      if (key) headers['X-API-Key'] = key;
      const response = await fetch('/api/v1/alerts/session', { method: 'POST', headers: headers });
      if (!response.ok) throw new Error(response.status === 401 ? 'API key rejected. Check the key and activate alerts again.' : 'Could not create a live-alert session (' + response.status + ').');
      const session = await response.json();
      const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const socket = new WebSocket(scheme + '//' + window.location.host + '/ws/alerts?token=' + encodeURIComponent(session.token));
      state.socket = socket;
      socket.onopen = () => {
        setConnection(true, 'LIVE CONNECTED');
        state.pingTimer = window.setInterval(() => {
          if (socket.readyState === WebSocket.OPEN) socket.send('ping');
        }, 25000);
      };
      socket.onmessage = (message) => {
        try { handleLiveEvent(JSON.parse(message.data)); } catch (error) { console.warn('Invalid live alert event:', error); }
      };
      socket.onerror = () => setConnection(false, 'CONNECTION ERROR');
      socket.onclose = () => {
        window.clearInterval(state.pingTimer);
        setConnection(false, state.alertEnabled ? 'RECONNECTING' : 'NOT CONNECTED');
        if (state.alertEnabled) state.reconnectTimer = window.setTimeout(connectLiveAlerts, 2200);
      };
    } catch (error) {
      setConnection(false, 'NOT CONNECTED');
      setMessage(error.message || 'Could not connect to the live alert stream.', true);
      state.reconnectTimer = window.setTimeout(connectLiveAlerts, 5000);
    }
  };

  const activateAlerts = async () => {
    state.alertEnabled = true;
    state.soundEnabled = true;
    $('sound-state').setAttribute('aria-pressed', 'true');
    $('sound-label').textContent = 'Sound on';
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass && !state.audioContext) state.audioContext = new AudioContextClass();
    if (state.audioContext && state.audioContext.state === 'suspended') await state.audioContext.resume();
    if ('Notification' in window && Notification.permission === 'default') {
      try { await Notification.requestPermission(); } catch (_) { /* In-app alerts remain available. */ }
    }
    $('activate-alerts').textContent = 'Live alerts active';
    $('activate-alerts').disabled = true;
    setMessage('Live alert stream is starting. Run a scan to test it.', false);
    connectLiveAlerts();
  };

  const scanTransactions = async () => {
    if (!state.rows.length) return;
    $('scan-button').disabled = true;
    $('scan-button').textContent = 'Scanning…';
    setMessage('Sending ' + state.rows.length + ' transactions to FinGuard…', false);
    try {
      const headers = { 'Content-Type': 'application/json' };
      const key = $('api-key').value.trim();
      if (key) headers['X-API-Key'] = key;
      const response = await fetch('/api/v1/transactions/scan', {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ transactions: state.rows })
      });
      const result = await response.json();
      if (!response.ok) {
        const detail = typeof result.detail === 'string' ? result.detail : 'Request rejected (' + response.status + ').';
        throw new Error(detail);
      }
      const liveSocketConnected = state.socket && state.socket.readyState === WebSocket.OPEN;
      if (!liveSocketConnected) {
        state.scanned += Number(result.transactions_analyzed || 0);
        state.flagged += Number(result.flagged_count || 0);
        state.high += result.alerts.filter((alert) => alert.fraud_risk_level === 'HIGH' || alert.trust_classification === 'Manipulated' || alert.explanation_verdict === 'UNTRUSTWORTHY').length;
      }
      renderResults(result);
      setMessage('Scan complete · ' + result.flagged_count + ' flagged for review · ' + result.analysis_type.replaceAll('_', ' '), false);
      if (!state.alertEnabled && result.flagged_count) {
        setMessage('Scan complete. Activate live alerts to receive a toast and sound for the next scan event.', false);
      }
    } catch (error) {
      setMessage(error.message || 'The scan failed. Check that the API is running.', true);
    } finally {
      $('scan-button').disabled = false;
      $('scan-button').innerHTML = 'Scan transactions <span aria-hidden="true">↗</span>';
    }
  };

  const setRows = (rows, label) => {
    if (!rows.length) {
      setMessage('No valid transactions found. Check the amount column and CSV headers.', true);
      return;
    }
    if (rows.length > 5000) throw new Error('The API accepts up to 5,000 transactions per scan.');
    state.rows = rows;
    $('file-title').textContent = label;
    $('file-subtitle').textContent = rows.length + ' valid transaction records ready';
    $('scan-button').disabled = false;
    setMessage(rows.length + ' transactions ready to scan.', false);
  };

  const loadFile = async (file) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setMessage('Please choose a .csv file.', true);
      return;
    }
    try {
      const text = await file.text();
      setRows(parseCSV(text), file.name);
    } catch (error) {
      setMessage(error.message || 'Could not read this CSV file.', true);
    }
  };

  $('drop-zone').addEventListener('click', (event) => {
    if (event.target !== $('csv-file')) $('csv-file').click();
  });
  $('drop-zone').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('csv-file').click(); }
  });
  $('csv-file').addEventListener('change', (event) => loadFile(event.target.files[0]));
  ['dragenter', 'dragover'].forEach((name) => $('drop-zone').addEventListener(name, (event) => {
    event.preventDefault(); $('drop-zone').classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((name) => $('drop-zone').addEventListener(name, (event) => {
    event.preventDefault(); $('drop-zone').classList.remove('dragging');
  }));
  $('drop-zone').addEventListener('drop', (event) => loadFile(event.dataTransfer.files[0]));
  $('scan-button').addEventListener('click', scanTransactions);
  $('sample-button').addEventListener('click', () => {
    const sample = [{
      id: 'DEMO-' + Date.now(),
      date: (() => {
        const local = new Date();
        local.setHours(2, 18, 0, 0);
        return local.getFullYear() + '-' + String(local.getMonth() + 1).padStart(2, '0') + '-' + String(local.getDate()).padStart(2, '0') + 'T02:18:00';
      })(),
      description: 'Unrecognised late-night bank transfer',
      amount: 27800,
      direction: 'debit',
      category: 'Transfer',
      method: 'UPI',
      receiver: 'sample-recipient',
      status: 'complete'
    }];
    setRows(sample, 'FinGuard sample transaction');
    setMessage('Sample loaded. This is synthetic demonstration data, not a real fraud event.', false);
  });
  $('activate-alerts').addEventListener('click', activateAlerts);
  $('sound-state').addEventListener('click', async () => {
    state.soundEnabled = !state.soundEnabled;
    if (state.soundEnabled) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass && !state.audioContext) state.audioContext = new AudioContextClass();
      if (state.audioContext && state.audioContext.state === 'suspended') await state.audioContext.resume();
    }
    $('sound-state').setAttribute('aria-pressed', String(state.soundEnabled));
    $('sound-label').textContent = state.soundEnabled ? 'Sound on' : 'Sound off';
  });

  // A conspicuous but low-cost gold dust field creates continuous visible motion.
  const canvas = $('particle-field');
  const context = canvas.getContext('2d');
  let particles = [];
  const resizeCanvas = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.75);
    canvas.width = Math.floor(window.innerWidth * ratio);
    canvas.height = Math.floor(window.innerHeight * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    particles = Array.from({ length: Math.min(92, Math.floor(window.innerWidth / 13)) }, () => ({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      radius: Math.random() * 1.8 + .35,
      speed: Math.random() * .42 + .10,
      drift: (Math.random() - .5) * .34,
      alpha: Math.random() * .54 + .13,
      phase: Math.random() * Math.PI * 2
    }));
  };
  let frame = 0;
  const drawParticles = () => {
    if (!context) return;
    context.clearRect(0, 0, window.innerWidth, window.innerHeight);
    frame += 1;
    particles.forEach((particle) => {
      particle.y -= particle.speed;
      particle.x += particle.drift + Math.sin(frame * .007 + particle.phase) * .16;
      if (particle.y < -5) { particle.y = window.innerHeight + 5; particle.x = Math.random() * window.innerWidth; }
      if (particle.x < -5) particle.x = window.innerWidth + 5;
      if (particle.x > window.innerWidth + 5) particle.x = -5;
      const pulse = .7 + .3 * Math.sin(frame * .025 + particle.phase);
      context.beginPath();
      context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
      context.fillStyle = 'rgba(238, 190, 96, ' + (particle.alpha * pulse) + ')';
      context.shadowBlur = 8;
      context.shadowColor = 'rgba(242, 188, 78, .75)';
      context.fill();
    });
    context.shadowBlur = 0;
    window.requestAnimationFrame(drawParticles);
  };
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) window.requestAnimationFrame(drawParticles);

  // Small neon sparks follow the exact click location, then fade out.
  document.addEventListener('click', (event) => {
    if (event.target.closest('button, a, input, .drop-zone')) {
      // The sparkle still appears over controls; this branch intentionally has no early return.
    }
    const colors = ['#5ce5e3', '#a98bff', '#f4c76b', '#88f0e0'];
    for (let index = 0; index < 7; index += 1) {
      const spark = document.createElement('i');
      spark.className = 'spark';
      const angle = (Math.PI * 2 * index / 7) + Math.random() * .28;
      const distance = 15 + Math.random() * 23;
      spark.style.left = event.clientX + 'px';
      spark.style.top = event.clientY + 'px';
      spark.style.color = colors[index % colors.length];
      spark.style.background = colors[index % colors.length];
      document.body.appendChild(spark);
      requestAnimationFrame(() => {
        spark.style.transform = 'translate(' + (Math.cos(angle) * distance) + 'px,' + (Math.sin(angle) * distance) + 'px) scale(.25)';
        spark.style.opacity = '0';
      });
      window.setTimeout(() => spark.remove(), 550);
    }
  });

  setConnection(false, 'NOT CONNECTED');
})();