const hostInput = document.getElementById('host-input');
const portInput = document.getElementById('port-input');
const connectBtn = document.getElementById('connect-btn');
const disconnectBtn = document.getElementById('disconnect-btn');
const statusBadge = document.getElementById('connection-status');
const telemetryFields = {
  temperature_c: document.getElementById('metric-temp'),
  speed_cm_s: document.getElementById('metric-speed'),
  battery_percent: document.getElementById('metric-battery'),
  power_percent: document.getElementById('metric-power'),
  steering_deg: document.getElementById('metric-steering'),
};
const commandButtons = document.querySelectorAll('[data-command]');
const logContainer = document.getElementById('log-container');

const MAX_LOG_ENTRIES = 250;
let isConnected = false;
let suppressLogs = false;
const currentTelemetry = {
  temperature_c: null,
  speed_cm_s: null,
  battery_percent: null,
  power_percent: null,
  steering_deg: null,
};

function applyDefaults() {
  const { host, port } = window.carApi.defaults;
  hostInput.value = host;
  portInput.value = port;
}

function setStatus(state, message) {
  statusBadge.textContent = message;
  statusBadge.classList.remove('status-connected', 'status-error', 'status-idle');
  if (state === 'connected') {
    statusBadge.classList.add('status-connected');
  } else if (state === 'error') {
    statusBadge.classList.add('status-error');
  } else {
    statusBadge.classList.add('status-idle');
  }
}

function setControlsEnabled(enabled) {
  commandButtons.forEach((btn) => {
    btn.disabled = !enabled;
  });
  disconnectBtn.disabled = !enabled;
  connectBtn.disabled = enabled;
}

function sanitiseNumber(value, unit) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return `-- ${unit}`;
  }
  const formatted = Number.parseFloat(value).toFixed(2);
  return `${formatted} ${unit}`;
}

function updateTelemetry(payload) {
  if (!payload) {
    return;
  }

  const keys = Object.keys(currentTelemetry);
  keys.forEach((key) => {
    if (Object.prototype.hasOwnProperty.call(payload, key)) {
      currentTelemetry[key] = payload[key];
    }
  });

  telemetryFields.temperature_c.textContent = sanitiseNumber(currentTelemetry.temperature_c, '°C');
  telemetryFields.speed_cm_s.textContent = sanitiseNumber(currentTelemetry.speed_cm_s, 'cm/s');
  telemetryFields.battery_percent.textContent = sanitiseNumber(currentTelemetry.battery_percent, '%');
  telemetryFields.power_percent.textContent = sanitiseNumber(currentTelemetry.power_percent, '%');
  telemetryFields.steering_deg.textContent = sanitiseNumber(currentTelemetry.steering_deg, '°');
}

function makeLogEntry(level, message, details) {
  const line = document.createElement('p');
  line.className = 'log-entry';
  const timestamp = new Date().toLocaleTimeString();
  line.innerHTML = `<span class="timestamp">[${timestamp}]</span><strong>${level.toUpperCase()}:</strong> ${message}`;
  if (details) {
    const detailEl = document.createElement('code');
    detailEl.textContent = ` ${JSON.stringify(details)}`;
    line.appendChild(detailEl);
  }
  logContainer.appendChild(line);

  while (logContainer.children.length > MAX_LOG_ENTRIES) {
    logContainer.removeChild(logContainer.firstChild);
  }
  logContainer.scrollTop = logContainer.scrollHeight;
}

function handleCommandClick(event) {
  const command = event.currentTarget.dataset.command;
  try {
    window.carApi.sendCommand(command);
    makeLogEntry('command', `Sent ${command}`);
  } catch (error) {
    makeLogEntry('error', `Failed to send ${command}`, { message: error.message });
  }
}

function registerListeners() {
  connectBtn.addEventListener('click', async () => {
    const host = hostInput.value.trim();
    const port = portInput.value.trim();

    if (!host || !port) {
      setStatus('error', 'Missing host or port');
      return;
    }

    setStatus('idle', 'Connecting…');
    suppressLogs = false;
    connectBtn.disabled = true;

    try {
      await window.carApi.connect(host, port);
      setStatus('connected', `Connected to ${host}:${port}`);
      isConnected = true;
      setControlsEnabled(true);
      makeLogEntry('info', 'Connection established', { host, port });
    } catch (err) {
      setStatus('error', 'Connection failed');
      connectBtn.disabled = false;
      isConnected = false;
      setControlsEnabled(false);
      makeLogEntry('error', 'Failed to connect', { message: err.message });
    }
  });

  disconnectBtn.addEventListener('click', () => {
    window.carApi.disconnect();
    makeLogEntry('info', 'Disconnect requested by user');
  });

  commandButtons.forEach((btn) => btn.addEventListener('click', handleCommandClick));

  const keyMap = new Map([
    ['ArrowUp', 'FWD'],
    ['w', 'FWD'],
    ['W', 'FWD'],
    ['ArrowDown', 'BWD'],
    ['s', 'BWD'],
    ['S', 'BWD'],
    ['ArrowLeft', 'LT'],
    ['a', 'LT'],
    ['A', 'LT'],
    ['ArrowRight', 'RT'],
    ['d', 'RT'],
    ['D', 'RT'],
    [' ', 'STOP'],
  ]);

  document.addEventListener('keydown', (event) => {
    if (!isConnected || event.repeat) {
      return;
    }
    const command = keyMap.get(event.key);
    if (command) {
      event.preventDefault();
      try {
        window.carApi.sendCommand(command);
        makeLogEntry('command', `Sent ${command} (keyboard)`);
      } catch (error) {
        makeLogEntry('error', `Failed to send ${command}`, { message: error.message });
      }
    }
  });

  window.carApi.on('connected', ({ host, port }) => {
    isConnected = true;
    setControlsEnabled(true);
    setStatus('connected', `Connected to ${host}:${port}`);
    makeLogEntry('info', 'Connected', { host, port });
  });

  window.carApi.on('disconnected', () => {
    isConnected = false;
    setControlsEnabled(false);
    connectBtn.disabled = false;
    if (!suppressLogs) {
      makeLogEntry('warn', 'Disconnected');
    }
    setStatus('idle', 'Disconnected');
  });

  window.carApi.on('telemetry', (payload) => {
    updateTelemetry(payload);
  });

  window.carApi.on('ack', (payload) => {
    updateTelemetry(payload);
    makeLogEntry('ack', `Command ${payload.command} acknowledged`, {
      status: payload.status,
      power: payload.power_percent,
      steering: payload.steering_deg,
    });
  });

  window.carApi.on('error', (error) => {
    setStatus('error', error.message || 'Socket error');
    makeLogEntry('error', 'Socket error', { message: error.message });
  });

  window.carApi.on('message', (payload) => {
    if (payload.type === 'raw') {
      makeLogEntry('warn', 'Non-JSON payload', { raw: payload.raw });
    }
  });

  window.addEventListener('beforeunload', () => {
    suppressLogs = true;
    window.carApi.disconnect();
  });
}

applyDefaults();
setControlsEnabled(false);
registerListeners();
setStatus('idle', 'Idle');
makeLogEntry('info', 'Ready. Configure the IP address of your Raspberry Pi and press Connect.');
