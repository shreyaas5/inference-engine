import * as http from 'node:http';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { getEngineManager, EngineManager } from './engineManager.js';

const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || '127.0.0.1';

const manager: EngineManager = getEngineManager();

function sendJson(res: http.ServerResponse, status: number, data: any) {
  const payload = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });
  res.end(payload);
}

function handleCors(req: http.IncomingMessage, res: http.ServerResponse): boolean {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    });
    res.end();
    return true;
  }
  return false;
}

function parseBody(req: http.IncomingMessage): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = '';
    req.on('data', (chunk) => {
      body += chunk;
      if (body.length > 5 * 1024 * 1024) {
        req.destroy();
        reject(new Error('Payload too large'));
      }
    });
    req.on('end', () => {
      if (!body.trim()) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(body));
      } catch (err) {
        reject(new Error('Invalid JSON'));
      }
    });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  if (handleCors(req, res)) return;

  const urlObj = new URL(req.url || '/', `http://${HOST}:${PORT}`);
  const pathname = urlObj.pathname;

  // 1. SSE Events endpoint
  if (pathname === '/api/events' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    });
    res.write('retry: 1000\n\n');

    // Send initial snapshot
    res.write(`event: snapshot\ndata: ${JSON.stringify({
      engine_connected: manager.engineConnected,
      engine_state: manager.engineState,
      model: manager.modelInfo,
      runtime: manager.runtimeInfo,
      available_models: manager.availableModels,
      logs: manager.logs.slice(-50),
    })}\n\n`);

    const listener = (data: any) => {
      const eventType = data.event || 'message';
      res.write(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    manager.on('sse_event', listener);

    req.on('close', () => {
      manager.off('sse_event', listener);
    });
    return;
  }

  // 2. Health endpoint
  if (pathname === '/api/health' && req.method === 'GET') {
    return sendJson(res, 200, manager.getHealth());
  }

  // 3. Model inspector endpoint
  if (pathname === '/api/model' && req.method === 'GET') {
    return sendJson(res, 200, {
      model: manager.modelInfo,
      available_models: manager.availableModels,
    });
  }

  // 4. System / Runtime telemetry endpoint
  if (pathname === '/api/system' && req.method === 'GET') {
    return sendJson(res, 200, {
      health: manager.getHealth(),
      runtime: manager.runtimeInfo,
      model: manager.modelInfo,
    });
  }

  // 5. Chat generation endpoint (can return direct completion or register for streaming)
  if (pathname === '/api/chat' && req.method === 'POST') {
    try {
      const body = await parseBody(req);
      const requestId = body.request_id || `req-${Date.now()}`;
      const streamMode = body.stream !== false;

      if (streamMode) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
        });
        res.write('retry: 500\n\n');

        const onEvent = (ev: any) => {
          const type = ev.event || 'message';
          res.write(`event: ${type}\ndata: ${JSON.stringify(ev)}\n\n`);
        };

        try {
          await manager.sendCommand({ ...body, cmd: 'chat', request_id: requestId }, onEvent);
          res.end();
        } catch (err: any) {
          res.write(`event: generation_error\ndata: ${JSON.stringify({ error: err?.message || String(err) })}\n\n`);
          res.end();
        }
      } else {
        const result = await manager.sendCommand({ ...body, cmd: 'chat', request_id: requestId });
        return sendJson(res, 200, result);
      }
    } catch (err: any) {
      return sendJson(res, 400, { error: err.message });
    }
    return;
  }

  // 6. Stop generation endpoint
  if (pathname === '/api/chat/stop' && req.method === 'POST') {
    const success = manager.requestStop();
    return sendJson(res, 200, { success });
  }

  // 7. Clear context / KV cache endpoint
  if (pathname === '/api/chat/clear' && req.method === 'POST') {
    try {
      const body = await parseBody(req);
      const result = await manager.sendCommand({ ...body, cmd: 'clear' });
      return sendJson(res, 200, result);
    } catch (err: any) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  // 8. Reconfigure model / threads / backend endpoint
  if (pathname === '/api/configure' && req.method === 'POST') {
    try {
      const body = await parseBody(req);
      const result = await manager.sendCommand({ ...body, cmd: 'configure' });
      return sendJson(res, 200, result);
    } catch (err: any) {
      return sendJson(res, 500, { error: err.message });
    }
  }

  // 9. Benchmark execution endpoint
  if (pathname === '/api/benchmark' && req.method === 'POST') {
    try {
      const body = await parseBody(req);
      const requestId = body.request_id || `bench-${Date.now()}`;
      const streamMode = body.stream !== false;

      if (streamMode) {
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'Access-Control-Allow-Origin': '*',
        });
        res.write('retry: 500\n\n');

        const onEvent = (ev: any) => {
          const type = ev.event || 'message';
          res.write(`event: ${type}\ndata: ${JSON.stringify(ev)}\n\n`);
        };

        try {
          const record = await manager.sendCommand({ ...body, cmd: 'benchmark', request_id: requestId }, onEvent);
          res.end();
        } catch (err: any) {
          res.write(`event: benchmark_error\ndata: ${JSON.stringify({ error: err?.message || String(err) })}\n\n`);
          res.end();
        }
      } else {
        const record = await manager.sendCommand({ ...body, cmd: 'benchmark', request_id: requestId });
        return sendJson(res, 200, record);
      }
    } catch (err: any) {
      return sendJson(res, 400, { error: err.message });
    }
    return;
  }

  // 10. Benchmark history endpoint
  if (pathname === '/api/benchmark/history' && req.method === 'GET') {
    return sendJson(res, 200, {
      history: manager.benchmarkHistory,
    });
  }

  // 11. Runtime logs endpoint
  if (pathname === '/api/logs' && req.method === 'GET') {
    return sendJson(res, 200, {
      logs: manager.logs,
    });
  }

  if (pathname === '/api/logs/clear' && req.method === 'POST') {
    manager.clearLogs();
    return sendJson(res, 200, { success: true });
  }

  // Serve static files in production if dist/ exists
  const distDir = path.resolve(import.meta.dirname, '../dist');
  if (fs.existsSync(distDir)) {
    let filePath = path.join(distDir, pathname === '/' ? 'index.html' : pathname);
    if (!fs.existsSync(filePath)) {
      filePath = path.join(distDir, 'index.html');
    }

    const ext = path.extname(filePath).toLowerCase();
    const mimeMap: Record<string, string> = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.css': 'text/css',
      '.json': 'application/json',
      '.png': 'image/png',
      '.svg': 'image/svg+xml',
      '.ico': 'image/x-icon',
      '.woff2': 'font/woff2',
    };

    const contentType = mimeMap[ext] || 'application/octet-stream';
    try {
      const content = fs.readFileSync(filePath);
      res.writeHead(200, {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      });
      res.end(content);
      return;
    } catch {
      // fallback
    }
  }

  sendJson(res, 404, { error: `Not found: ${pathname}` });
});

server.on('error', (err: any) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`[BIOS V2 Studio API] Port ${PORT} is already in use.`);
    console.error(`To terminate any existing studio process on port ${PORT}, run:`);
    console.error(`  lsof -ti :${PORT} | xargs kill -9`);
    console.error(`Or set a different port with: PORT=3001 npm start`);
  } else {
    console.error(`[BIOS V2 Studio API] Server error:`, err);
  }
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  console.log(`[BIOS V2 Studio API] Server listening on http://${HOST}:${PORT}`);
  manager.addLog('INFO', 'studio-api', `Inference Studio API server listening on http://${HOST}:${PORT}`);
});

