"""Run the TUI headlessly against the real API.

Textual's ``run_test()`` pilot replaces the terminal driver, so the interface
runs for real -- including its agent subprocess -- without a TTY to attach to.
That makes it usable as a smoke test on a machine where nobody can press keys.

    uv pip install textual        # tui.py declares it as an inline script dep
    uv run python -m bench.tui_run
    uv run python -m bench.tui_run --prompt "read README.md and summarise it"

The default prompt is read-only on purpose: it exercises the feed, the tool
cards and the worker protocol without opening an approval modal that nothing is
there to answer.
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_harness.config import normalize_no_proxy  # noqa: E402

normalize_no_proxy()

DEFAULT_PROMPT = ('Use glob_file to list the Python files under src/mini_harness, '
                  'then read src/mini_harness/budget.py and tell me how many lines it has. '
                  'Keep the answer to two sentences.')


def load_tui():
    try:
        import tui
    except ImportError as error:                 # pragma: no cover - environment dependent
        raise SystemExit(
            f'the TUI needs textual ({error}).\n'
            'Install it with:  uv pip install textual\n'
            'or run the entry point directly:  uv run --script tui.py --help')
    return tui


async def drive(prompt: str, timeout: float, sessions: Path, size: tuple,
                demo: bool = False) -> int:
    tui = load_tui()
    shutil.rmtree(sessions, ignore_errors=True)
    sessions.mkdir(parents=True, exist_ok=True)
    app = tui.MiniHarness(demo=demo, sessions_dir=sessions)

    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        print(f'mounted  : phase={app.phase!r}')
        app.query_one('#prompt', tui.Input).value = prompt
        await pilot.press('enter')

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while app.busy:
            if loop.time() > deadline:
                print(f'timed out after {timeout:.0f}s; cancelling')
                await app.action_stop()
                break
            await pilot.pause(0.25)

        await pilot.pause(0.5)
        await app.action_quit_cleanly()
        await pilot.pause()

    print(f'phase    : {app.phase!r}')
    print(f'tokens   : prompt={app.prompt_tokens} output={app.output_tokens} calls={app.calls}')
    print('--- transcript ---')
    for entry in app.session.get('events', []):
        kind = entry.get('kind')
        if kind == 'user':
            print(f'[user] {str(entry.get("text", ""))[:100]}')
        elif kind == 'assistant':
            print(f'[assistant] {str(entry.get("text", "")).strip()[:600]}')
        elif kind == 'tool':
            print(f'[tool] {entry.get("name")} status={entry.get("status")} '
                  f'args={str(entry.get("arguments"))[:80]}')
        else:
            print(f'[{kind}] {str(entry)[:160]}')
    return 0 if app.phase == 'COMPLETED' else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--timeout', type=float, default=300.0)
    parser.add_argument('--sessions', default=str(ROOT / '.experiments' / 'tui-sessions'))
    parser.add_argument('--size', default='100x32', help='terminal size as WxH')
    parser.add_argument('--demo', action='store_true',
                        help='offline simulation; needs no API key and makes no model calls')
    args = parser.parse_args()

    width, _, height = args.size.partition('x')
    size = (int(width), int(height or 32))
    return asyncio.run(drive(args.prompt, args.timeout, Path(args.sessions), size, args.demo))


if __name__ == '__main__':
    raise SystemExit(main())
