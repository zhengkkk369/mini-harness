"""A minimal stdio MCP server, used to test the bridge without a network.

Run by tests/mcp_tests, not collected by pytest (the file name does not start
with test_). Flags exist only to make failure modes reachable:

    --bad-handshake   answer initialize with a JSON-RPC error
    --noisy           emit a non-JSON line, a notification, and a server request
    --silent          never answer anything
    --binary          emit a line containing bytes that are not valid UTF-8
"""

import json
import sys

# The protocol is UTF-8 on stdout. A Python server on a non-UTF-8 console has to
# say so, or its own printing dies on the first non-ASCII character it sends.
sys.stdout.reconfigure(encoding='utf-8', newline='\n')

TOOLS = [
    {
        'name': 'echo',
        'description': 'Echo the text back.',
        'inputSchema': {'type': 'object',
                        'properties': {'text': {'type': 'string', 'description': 'what to echo'}},
                        'required': ['text']},
    },
    {
        'name': 'add',
        'description': 'Add two integers.',
        'inputSchema': {'type': 'object',
                        'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                        'required': ['a', 'b']},
    },
    {
        'name': 'optional',
        'description': 'Everything is optional.',
        'inputSchema': {'type': 'object', 'properties': {'note': {'type': 'string'}}},
    },
    {
        'name': 'fail',
        'description': 'Always reports an error.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'slow',
        'description': 'Never answers.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'no schema',
        'description': 'A name that is not registry safe.',
        'inputSchema': {'type': 'object', 'properties': {'x': {'type': 'string'}}},
    },
]


def send(payload: dict) -> None:
    # ensure_ascii=False on purpose: non-ASCII must travel as real UTF-8 bytes,
    # otherwise the reader's encoding would never be exercised.
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')
    sys.stdout.flush()


def result(request_id, value) -> None:
    send({'jsonrpc': '2.0', 'id': request_id, 'result': value})


def error(request_id, code, message) -> None:
    send({'jsonrpc': '2.0', 'id': request_id, 'error': {'code': code, 'message': message}})


def call_tool(request_id, name, arguments) -> None:
    if name == 'echo':
        result(request_id, {'content': [{'type': 'text', 'text': f"echo: {arguments.get('text', '')}"}]})
    elif name == 'add':
        result(request_id, {'content': [{'type': 'text',
                                         'text': str(arguments.get('a', 0) + arguments.get('b', 0))}]})
    elif name == 'optional':
        result(request_id, {'content': [{'type': 'text', 'text': f"note={arguments.get('note')}"},
                                        {'type': 'image', 'data': 'ignored'}]})
    elif name == 'fail':
        result(request_id, {'content': [{'type': 'text', 'text': 'boom'}], 'isError': True})
    elif name == 'slow':
        pass
    else:
        error(request_id, -32602, f'unknown tool {name!r}')


def main() -> int:
    flags = set(sys.argv[1:])
    if '--binary' in flags:
        # A server may emit bytes that are not valid UTF-8; a text-mode pipe
        # decoded with the locale encoding would die on this.
        sys.stdout.flush()
        sys.stdout.buffer.write(b'{"jsonrpc":"2.0","method":"x","params":"\xff\xfe"}\n')
        sys.stdout.buffer.flush()
    if '--noisy' in flags:
        sys.stdout.write('this line is not json\n')
        sys.stdout.flush()
        send({'jsonrpc': '2.0', 'method': 'notifications/message',
              'params': {'level': 'info', 'data': 'hello'}})
        send({'jsonrpc': '2.0', 'id': 'server-side-1', 'method': 'roots/list'})
    if '--silent' in flags:
        for _line in sys.stdin:
            pass
        return 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method = message.get('method')
        request_id = message.get('id')
        if method == 'initialize':
            if '--bad-handshake' in flags:
                error(request_id, -32000, 'no thanks')
            else:
                result(request_id, {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {'tools': {}},
                    'serverInfo': {'name': 'stub', 'version': '1'},
                })
        elif method == 'notifications/initialized':
            continue
        elif method == 'tools/list':
            result(request_id, {'tools': TOOLS})
        elif method == 'tools/call':
            params = message.get('params') or {}
            call_tool(request_id, params.get('name'), params.get('arguments') or {})
        elif request_id is not None:
            error(request_id, -32601, f'unknown method {method!r}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
