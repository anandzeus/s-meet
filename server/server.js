import cors from 'cors';
import express from 'express';
import http from 'http';
import { randomUUID } from 'crypto';
import { WebSocketServer, WebSocket } from 'ws';

const PORT = process.env.PORT || 3001;

const app = express();
app.use(cors());

app.get('/health', (req, res) => {
  res.json({ ok: true });
});

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

const rooms = new Map();

const ensureRoom = (roomId) => {
  if (!rooms.has(roomId)) {
    rooms.set(roomId, new Set());
  }
  return rooms.get(roomId);
};

const broadcast = (roomId, payload, except) => {
  const room = rooms.get(roomId);
  if (!room) {
    return;
  }
  const data = JSON.stringify(payload);
  for (const client of room) {
    if (client !== except && client.readyState === WebSocket.OPEN) {
      client.send(data);
    }
  }
};

const removeClient = (ws, client) => {
  if (!client.roomId) {
    return;
  }
  const room = rooms.get(client.roomId);
  if (!room) {
    client.roomId = null;
    return;
  }
  room.delete(ws);
  broadcast(client.roomId, { type: 'peer-left', id: client.id, name: client.name }, ws);
  if (room.size === 0) {
    rooms.delete(client.roomId);
  }
  client.roomId = null;
};

wss.on('connection', (ws) => {
  const client = {
    id: randomUUID(),
    roomId: null,
    name: 'Guest'
  };

  ws.on('message', (raw) => {
    let message;
    try {
      message = JSON.parse(raw.toString());
    } catch (error) {
      return;
    }

    if (message.type === 'join') {
      const roomId = String(message.roomId || '').trim();
      if (!roomId) {
        ws.send(JSON.stringify({ type: 'error', message: 'roomId is required' }));
        return;
      }

      if (client.roomId && client.roomId !== roomId) {
        removeClient(ws, client);
      }

      client.roomId = roomId;
      client.name = String(message.name || 'Guest').trim() || 'Guest';
      ws.clientId = client.id;

      const room = ensureRoom(roomId);
      room.add(ws);

      ws.send(
        JSON.stringify({
          type: 'room-info',
          roomId,
          id: client.id,
          peers: room.size
        })
      );

      broadcast(roomId, { type: 'peer-joined', id: client.id, name: client.name }, ws);
      return;
    }

    if (!client.roomId) {
      ws.send(JSON.stringify({ type: 'error', message: 'join a room first' }));
      return;
    }

    if (message.type === 'leave') {
      removeClient(ws, client);
      return;
    }

    if (message.target) {
      const room = rooms.get(client.roomId);
      if (room) {
        for (const peerWs of room) {
          if (peerWs.clientId === message.target && peerWs.readyState === WebSocket.OPEN) {
            peerWs.send(
              JSON.stringify({
                ...message,
                from: { id: client.id, name: client.name }
              })
            );
            break;
          }
        }
      }
      return;
    }

    broadcast(
      client.roomId,
      {
        ...message,
        from: { id: client.id, name: client.name }
      },
      ws
    );
  });

  ws.on('close', () => {
    removeClient(ws, client);
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Signaling server listening on http://0.0.0.0:${PORT}`);
});