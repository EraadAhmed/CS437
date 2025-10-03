const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const net = require('node:net');
const path = require('path');

const isMac = process.platform === 'darwin';
let mainWindow;

const carState = {
  socket: null,
  buffer: '',
  host: null,
  port: null,
  connecting: false,
};

function sendToRenderer(channel, payload = {}) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

function resetSocketState({ notify = true } = {}) {
  if (carState.socket) {
    carState.socket.removeAllListeners();
    try {
      carState.socket.end();
    } catch (err) {
      // ignore during teardown
    }
    try {
      carState.socket.destroy();
    } catch (err) {
      // ignore during teardown
    }
    carState.socket = null;
  }
  carState.buffer = '';
  carState.host = null;
  carState.port = null;
  carState.connecting = false;
  if (notify) {
    sendToRenderer('car:disconnected');
  }
}

function forwardPayload(payload) {
  sendToRenderer('car:message', payload);
  if (!payload || typeof payload.type !== 'string') {
    return;
  }

  switch (payload.type) {
    case 'telemetry':
      sendToRenderer('car:telemetry', payload);
      break;
    case 'ack':
      sendToRenderer('car:ack', payload);
      break;
    case 'error':
      sendToRenderer('car:command-error', payload);
      break;
    default:
      break;
  }
}

function handleSocketData(chunk) {
  carState.buffer += chunk.toString('utf8');

  while (true) {
    const match = carState.buffer.match(/^(.*?)[\r\n]+/);
    if (!match) {
      break;
    }

    const line = match[1].trim();
    carState.buffer = carState.buffer.slice(match[0].length);

    if (!line) {
      continue;
    }

    try {
      const payload = JSON.parse(line);
      forwardPayload(payload);
    } catch (err) {
      sendToRenderer('car:message', {
        type: 'raw',
        raw: line,
        error: err.message,
      });
    }
  }
}

function registerIpcHandlers() {
  ipcMain.handle('car:connect', (_event, args) => {
    const host = args?.host?.toString().trim();
    const port = Number.parseInt(args?.port, 10);

    if (!host || Number.isNaN(port)) {
      throw new Error('Invalid host or port');
    }
    if (carState.socket || carState.connecting) {
      throw new Error('Already connected');
    }

    return new Promise((resolve, reject) => {
      carState.connecting = true;
      const socket = net.createConnection({ host, port });
      let settled = false;

      const handleError = (err) => {
        if (!settled) {
          settled = true;
          carState.connecting = false;
          socket.destroy();
          reject(new Error(err.message || 'Connection failed'));
        } else {
          sendToRenderer('car:error', { message: err.message });
          resetSocketState({ notify: true });
        }
      };

      socket.once('error', handleError);

      socket.once('connect', () => {
        socket.removeListener('error', handleError);
        carState.socket = socket;
        carState.host = host;
        carState.port = port;
        carState.connecting = false;
        settled = true;

        socket.setKeepAlive(true, 60_000);
        socket.on('data', handleSocketData);
        socket.on('error', (err) => {
          sendToRenderer('car:error', { message: err.message });
        });
        socket.on('close', () => {
          resetSocketState({ notify: true });
        });

        sendToRenderer('car:connected', { host, port });
        resolve({ host, port });
      });

      socket.on('error', handleError);
    });
  });

  ipcMain.handle('car:disconnect', () => {
    resetSocketState({ notify: true });
    return true;
  });

  ipcMain.handle('car:send', (_event, command) => {
    const trimmed = command?.toString().trim();
    if (!carState.socket) {
      throw new Error('Not connected');
    }
    if (!trimmed) {
      return false;
    }
    return new Promise((resolve, reject) => {
      carState.socket.write(`${trimmed}\n`, (err) => {
        if (err) {
          reject(new Error(err.message || 'Failed to send command'));
          return;
        }
        resolve(true);
      });
    });
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 900,
    minWidth: 900,
    minHeight: 700,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'public', 'index.html'));

  if (process.env.NODE_ENV === 'development') {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(() => {
  createWindow();
  registerIpcHandlers();

  const template = [
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forcereload' },
        { type: 'separator' },
        { role: 'toggledevtools' },
        { type: 'separator' },
        { role: 'resetzoom' },
        { role: 'zoomin' },
        { role: 'zoomout' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'zoom' },
        ...(isMac ? [{ type: 'separator' }, { role: 'front' }, { type: 'separator' }, { role: 'window' }] : [{ role: 'close' }]),
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (!isMac) {
    app.quit();
  }
});

app.on('quit', () => {
  resetSocketState({ notify: false });
});
