"""Bridge tools from an MCP server into the built-in tool registry.

A stdio MCP server is a subprocess speaking JSON-RPC 2.0, one JSON object per
line. This client performs the handshake, asks for the tool list, and wraps each
remote tool in a ToolDefinition so the model sees it exactly like a local one --
same schema, same registry, same policy, same tracing.

Two deliberate choices:

* The advertised schema is the server's own, not one re-derived from a Pydantic
  model, because a round trip through a model would flatten detail the server
  published. Validation still goes through a generated model, so an obviously
  wrong call is refused before it reaches the server.
* Remote tools default to ``risky``, so they ask for approval like the local
  shell does. A server can do anything.
"""

import json
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time

from dataclasses import dataclass, field
from typing import Any

from pydantic import ConfigDict, Field, create_model

from mini_harness.config import CONFIG
from mini_harness.trace import TRACE
from mini_harness.tool.box import ToolDefinition

PROTOCOL_VERSION = '2024-11-05'
CLIENT_INFO = {'name': 'mini-harness', 'version': '0.1.0'}
# A server may page its tool list. This bounds a server that never stops.
MAX_PAGES = 50
NAME_SAFE = re.compile(r'[^A-Za-z0-9_-]')
JSON_TYPES = {
    'string': str,
    'integer': int,
    'number': float,
    'boolean': bool,
    'array': list,
    'object': dict,
    'null': type(None),
}

class MCPError(RuntimeError):
    """A server failed, refused a call, or stopped answering."""

# Pushed by the reader when the pipe ends, so a waiting request fails at once
# instead of sitting out its whole timeout.
CLOSED = object()

def parse_servers(specs) -> list:
    """Turn ``name=command line`` specs into (name, argv) pairs."""
    servers = []
    for spec in specs or ():
        if not isinstance(spec, str) or '=' not in spec:
            raise ValueError(f'[mcp]: a server spec must look like name=command, got {spec!r}')
        name, _, command = spec.partition('=')
        name = name.strip()
        try:
            argv = shlex.split(command)
        except ValueError as error:
            raise ValueError(f'[mcp]: cannot parse the command for {name!r}: {error}') from None
        if not name or not argv:
            raise ValueError(f'[mcp]: a server spec must look like name=command, got {spec!r}')
        servers.append((name, argv))
    return servers

def tool_name(server: str, tool: str) -> str:
    """A registry-safe name that says which server the tool came from."""
    return NAME_SAFE.sub('_', f'{server}__{tool}')[:64]

def _annotation(spec: dict):
    if not isinstance(spec, dict):
        return Any
    declared = spec.get('type')
    if isinstance(declared, list):
        declared = next((item for item in declared if item != 'null'), None)
    return JSON_TYPES.get(declared, Any)

def model_from_schema(name: str, schema: dict) -> type:
    """A validating model for a JSON Schema object.

    Only what the schema declares is required; unknown types fall back to Any so
    a server cannot make a call impossible by publishing something exotic. Extra
    keys are allowed because a converted schema may not describe everything the
    server accepts.
    """
    properties = schema.get('properties') if isinstance(schema, dict) else None
    properties = properties if isinstance(properties, dict) else {}
    required = schema.get('required') if isinstance(schema, dict) else None
    required = set(required) if isinstance(required, list) else set()

    fields = {}
    for prop, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        annotation = _annotation(spec)
        description = spec.get('description')
        if prop in required:
            fields[prop] = (annotation, Field(..., description=description))
        else:
            fields[prop] = (annotation|None, Field(None, description=description))
    return create_model(name, __config__=ConfigDict(extra='allow'), **fields)

def _render(result: dict) -> str:
    blocks = result.get('content') if isinstance(result, dict) else None
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get('type') == 'text':
            parts.append(str(block.get('text', '')))
        else:
            parts.append(json.dumps(block, ensure_ascii=False, default=str))
    return '\n'.join(part for part in parts if part) or 'no result'

class MCPClient:
    """One stdio MCP server, started on demand and shut down explicitly."""

    def __init__(self, name: str, command: list, timeout: float = 30.0, env: dict|None = None,
                 cwd: str|None = None) -> None:
        self.name = name
        self.command = list(command)
        self.timeout = timeout
        self.env = env
        self.cwd = cwd
        self.process = None
        self.closed = False
        self.pages = 0
        self.notifications: list = []
        self.noise: list = []
        self._inbox: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._seq = 0
        self._reader = None
        return

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> 'MCPClient':
        command = list(self.command)
        resolved = shutil.which(command[0]) if command else None
        if resolved:
            # On Windows an entry point such as npx is really npx.CMD, and
            # CreateProcess will not find it from the bare name.
            command[0] = resolved
        self.closed = False
        self.process = subprocess.Popen(
            command, stdin = subprocess.PIPE, stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL, text = True, bufsize = 1,
            # A server is free to send UTF-8, and text mode would decode it with
            # the machine's locale encoding, so on a non-UTF-8 console one odd
            # byte would kill the reader and every later call would time out.
            encoding = 'utf-8', errors = 'replace',
            env = self.env, cwd = self.cwd)
        self._reader = threading.Thread(target = self._pump, name = f'mcp-{self.name}', daemon = True)
        self._reader.start()
        result = self._request('initialize', {
            'protocolVersion': PROTOCOL_VERSION,
            'capabilities': {},
            'clientInfo': CLIENT_INFO,
        })
        self.server_info = result.get('serverInfo') if isinstance(result, dict) else None
        self._notify('notifications/initialized', {})
        return self

    def close(self) -> None:
        if self.process is None:
            self.closed = True
            return
        self.closed = True
        process, self.process = self.process, None
        try:
            if process.stdin:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.terminate()
            process.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout = 5)
        except OSError:
            pass
        if self._reader is not None:
            self._reader.join(timeout = 1)
        return

    def __enter__(self) -> 'MCPClient':
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # ------------------------------------------------------------------ transport

    def _pump(self) -> None:
        """Read replies off the pipe; never let a bad line kill the server link."""
        stream = self.process.stdout if self.process else None
        try:
            if stream is None:
                return
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self.noise.append(line[:200])
                    continue
                if not isinstance(message, dict):
                    self.noise.append(line[:200])
                    continue
                if 'method' in message and 'id' in message:
                    # server -> client request: we support none of them yet
                    self._reply_unsupported(message.get('id'))
                    continue
                if 'id' in message:
                    self._inbox.put(message)
                    continue
                self.notifications.append(message)
        except (OSError, UnicodeDecodeError) as error:
            self.noise.append(f'reader stopped: {type(error).__name__}: {error}')
        finally:
            # Wake anyone waiting rather than leaving them to time out.
            self.closed = True
            self._inbox.put(CLOSED)
        return

    def _send(self, message: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPError(f'{self.name}: the server is not running')
        try:
            self.process.stdin.write(json.dumps(message) + '\n')
            self.process.stdin.flush()
        except (OSError, ValueError) as error:
            raise MCPError(f'{self.name}: cannot write to the server: {error}') from None
        return

    def _reply_unsupported(self, request_id) -> None:
        try:
            self._send({'jsonrpc': '2.0', 'id': request_id,
                        'error': {'code': -32601, 'message': 'not supported by mini-harness'}})
        except MCPError:
            pass
        return

    def _notify(self, method: str, params: dict) -> None:
        self._send({'jsonrpc': '2.0', 'method': method, 'params': params})
        return

    def _request(self, method: str, params: dict|None = None, timeout: float|None = None):
        """One request at a time, so replies cannot be matched to the wrong call."""
        with self._lock:
            if self.closed:
                raise MCPError(f'{self.name}: the server is not running any more')
            self._seq += 1
            request_id = self._seq
            self._send({'jsonrpc': '2.0', 'id': request_id, 'method': method,
                        'params': params or {}})
            deadline = time.time() + (self.timeout if timeout is None else timeout)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise MCPError(f'{self.name}: timed out waiting for {method}')
                try:
                    message = self._inbox.get(timeout = remaining)
                except queue.Empty:
                    raise MCPError(f'{self.name}: timed out waiting for {method}') from None
                if message is CLOSED:
                    raise MCPError(f'{self.name}: the server is not running any more')
                if message.get('id') != request_id:
                    continue
                if 'error' in message:
                    error = message['error'] or {}
                    raise MCPError(f"{self.name}: {method} failed: "
                                   f"{error.get('message', error)}")
                return message.get('result')

    # ------------------------------------------------------------------ tools

    def list_tools(self) -> list:
        """Every tool the server offers, following pagination cursors.

        The spec lets a server answer with one page and a ``nextCursor``. A
        client that ignores the cursor silently sees a subset of the tools, which
        looks like a server that simply has fewer of them.
        """
        tools, cursor, pages = [], None, 0
        while True:
            result = self._request('tools/list', {'cursor': cursor} if cursor else {})
            if not isinstance(result, dict):
                break
            tools.extend(tool for tool in (result.get('tools') or [])
                         if isinstance(tool, dict) and tool.get('name'))
            pages += 1
            cursor = result.get('nextCursor')
            if not cursor:
                break
            if pages >= MAX_PAGES:
                self.noise.append(f'tools/list still had a cursor after {MAX_PAGES} pages')
                break
        self.pages = pages
        return tools

    def call_tool(self, tool: str, arguments: dict) -> str:
        start = time.time()
        try:
            result = self._request('tools/call', {'name': tool, 'arguments': arguments})
        except MCPError as error:
            TRACE.emit('mcp_call', server = self.name, tool = tool, ok = False,
                       seconds = round(time.time() - start, 4), error = str(error)[:200])
            raise
        text = _render(result if isinstance(result, dict) else {})
        if isinstance(result, dict) and result.get('isError'):
            TRACE.emit('mcp_call', server = self.name, tool = tool, ok = False,
                       seconds = round(time.time() - start, 4), chars = len(text))
            raise MCPError(text)
        TRACE.emit('mcp_call', server = self.name, tool = tool, ok = True,
                   seconds = round(time.time() - start, 4), chars = len(text))
        return text

    def definition(self, tool: dict, risky: bool = True) -> ToolDefinition:
        """Wrap one remote tool so the executor can dispatch it."""
        remote = str(tool.get('name'))
        name = tool_name(self.name, remote)
        schema = tool.get('inputSchema')
        if not isinstance(schema, dict) or schema.get('type') != 'object':
            schema = {'type': 'object', 'properties': {}}
        model = model_from_schema(f'{name}_input', schema)
        description = str(tool.get('description') or f'{remote} on the {self.name} MCP server')
        client = self

        def invoke(args, cfg = None):
            return client.call_tool(remote, args.model_dump(exclude_none = True))

        return ToolDefinition(name = name, description = description, parameters = model,
                              function = invoke, risky = risky, schema = schema)

@dataclass
class MCPBridge:
    """Every configured server, and the tools they contributed."""

    cfg: object = CONFIG
    clients: list = field(default_factory = list)
    definitions: list = field(default_factory = list)
    failures: dict = field(default_factory = dict)

    def start(self) -> 'MCPBridge':
        for name, argv in parse_servers(self.cfg.mcp_servers):
            client = MCPClient(name, argv, timeout = self.cfg.mcp_timeout, env = self.cfg.bash_env)
            try:
                client.start()
                tools = client.list_tools()
            except (MCPError, OSError) as error:
                # One broken server must not stop the run.
                self.failures[name] = f'{type(error).__name__}: {error}'
                print(f'[mcp]: {name} unavailable: {self.failures[name]}')
                TRACE.emit('mcp_error', server = name, error = type(error).__name__,
                           message = str(error)[:200])
                client.close()
                continue
            self.clients.append(client)
            self.definitions.extend(client.definition(tool, risky = self.cfg.mcp_risky)
                                    for tool in tools)
            TRACE.emit('mcp_server', server = name, tools = [t.get('name') for t in tools],
                       pages = getattr(client, 'pages', 1),
                       server_info = getattr(client, 'server_info', None))
        return self

    def close(self) -> None:
        for client in self.clients:
            client.close()
        self.clients, self.definitions = [], []
        return

    def __enter__(self) -> 'MCPBridge':
        return self.start()

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
