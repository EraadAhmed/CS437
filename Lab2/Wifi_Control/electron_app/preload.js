const { contextBridge, ipcRenderer } = require('electron');

const DEFAULT_HOST = process.env.CAR_HOST || '192.168.125.22';
const DEFAULT_PORT = Number.parseInt(process.env.CAR_PORT || '65432', 10);

const channelMap = {
  connected: 'car:connected',
  disconnected: 'car:disconnected',
  telemetry: 'car:telemetry',
  ack: 'car:ack',
  'command-error': 'car:command-error',
  error: 'car:error',
  message: 'car:message',
  raw: 'car:raw',
};

const listenerRegistry = new Map();

function resolveChannel(event) {
  return channelMap[event] || event;
}

function on(event, handler) {
  const channel = resolveChannel(event);
  const wrapped = (_ipcEvent, payload) => handler(payload);
  ipcRenderer.on(channel, wrapped);

  const subscriptions = listenerRegistry.get(handler) || [];
  subscriptions.push({ channel, wrapped });
  listenerRegistry.set(handler, subscriptions);

  return () => {
    ipcRenderer.removeListener(channel, wrapped);
  };
}

function once(event, handler) {
  const channel = resolveChannel(event);
  const wrapped = (_ipcEvent, payload) => handler(payload);
  ipcRenderer.once(channel, wrapped);
}

function removeListener(event, handler) {
  const channel = resolveChannel(event);
  const subscriptions = listenerRegistry.get(handler);

  if (!subscriptions) {
    return;
  }

  for (const subscription of subscriptions) {
    if (subscription.channel === channel) {
      ipcRenderer.removeListener(channel, subscription.wrapped);
    }
  }

  listenerRegistry.delete(handler);
}

contextBridge.exposeInMainWorld('carApi', {
  defaults: { host: DEFAULT_HOST, port: DEFAULT_PORT },
  connect: (host, port) => ipcRenderer.invoke('car:connect', { host, port }),
  disconnect: () => ipcRenderer.invoke('car:disconnect'),
  sendCommand: (command) => ipcRenderer.invoke('car:send', command),
  on,
  once,
  removeListener,
});
