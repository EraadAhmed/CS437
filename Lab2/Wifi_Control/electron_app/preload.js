const { contextBridge } = require('electron');
const net = require('net');
const { EventEmitter } = require('events');

const DEFAULT_HOST = process.env.CAR_HOST || '192.168.125.22';
const DEFAULT_PORT = Number.parseInt(process.env.CAR_PORT || '65432', 10);

const emitter = new EventEmitter();
let socket = null;
let buffer = '';

function ensureSocket() {
  if (!socket) {
    const error = new Error('Socket is not connected');
    error.code = 'NOT_CONNECTED';
    throw error;
  }
}

function resetSocketState() {
  buffer = '';
  socket = null;
}

function parseIncoming(chunk) {
  buffer += chunk;
  let lineBreakIndex = buffer.search(/[\r\n]/);

  while (lineBreakIndex !== -1) {
    const line = buffer.slice(0, lineBreakIndex).trim();
    buffer = buffer.slice(lineBreakIndex + 1);

    if (line.length > 0) {
      let payload = null;
      try {
        payload = JSON.parse(line);
      } catch (err) {
        emitter.emit('message', { type: 'raw', raw: line, error: err.message });
        lineBreakIndex = buffer.search(/[\r\n]/);
        continue;
      }

      emitter.emit('message', payload);
      emitter.emit(payload.type || 'unknown', payload);
    }

    lineBreakIndex = buffer.search(/[\r\n]/);
  }
}

async function connect(host, port) {
  if (socket) {
    throw new Error('Already connected');
  }

  const resolvedHost = host || DEFAULT_HOST;
  const resolvedPort = Number.parseInt(port, 10) || DEFAULT_PORT;

  return new Promise((resolve, reject) => {
    const client = new net.Socket();
    let resolved = false;

    const handleError = (err) => {
      emitter.emit('error', err);
      client.destroy();
      if (!resolved) {
        resolved = true;
        reject(err);
      }
    };

    client.on('data', (chunk) => {
      parseIncoming(chunk.toString('utf8'));
    });

    client.on('error', handleError);

    client.on('close', () => {
      emitter.emit('disconnected');
      resetSocketState();
    });

    client.connect(resolvedPort, resolvedHost, () => {
      socket = client;
      socket.setKeepAlive(true, 60000);
      emitter.emit('connected', { host: resolvedHost, port: resolvedPort });
      if (!resolved) {
        resolved = true;
        resolve({ host: resolvedHost, port: resolvedPort });
      }
    });
  });
}

function disconnect() {
  if (socket) {
    socket.end();
    socket.destroy();
    resetSocketState();
  }
}

function sendCommand(command) {
  ensureSocket();
  const trimmed = (command || '').trim();
  if (!trimmed) {
    return;
  }
  socket.write(`${trimmed}\n`);
}

contextBridge.exposeInMainWorld('carApi', {
  defaults: { host: DEFAULT_HOST, port: DEFAULT_PORT },
  connect,
  disconnect,
  sendCommand,
  on: (event, handler) => {
    emitter.on(event, handler);
    return () => emitter.off(event, handler);
  },
  once: (event, handler) => emitter.once(event, handler),
  removeListener: (event, handler) => emitter.off(event, handler),
});
