"""Skills: procedures the agent can load on demand.

A skill is a Markdown file with a small header: a description, and optionally the
tools the procedure needs. The description goes into the system prompt, so the
agent knows the skill exists and what it is for; the body stays out of context
until the agent asks for it.

That split is the whole point. A skill is a document, not a capability: loading
one adds text and can pin tools the run already had, but it grants nothing. A
skill that names `run_bash` still gets the approval prompt, and one that names a
tool which does not exist is reported rather than quietly ignored.

The format is deliberately not YAML. This harness takes no dependencies beyond
the model client, so the header is `key: value` lines between two `---` rules:

    ---
    name: verify-a-change
    description: Run the tests that cover an edit, and report what they proved
    tools: read_file, run_bash
    ---

    Steps...
"""

import re

from dataclasses import dataclass, field
from pathlib import Path

SKILL_FILE = 'SKILL.md'
HEADER = '---'
MAX_INDEX_CHARS = 2000

def _split(text: str) -> tuple[dict, str]:
    """(header, body) for one skill file, or ({}, text) when there is no header."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != HEADER:
        return {}, text.strip()
    for index in range(1, len(lines)):
        if lines[index].strip() == HEADER:
            return _parse_header(lines[1:index]), '\n'.join(lines[index + 1:]).strip()
    return {}, text.strip()

def _parse_header(lines: list) -> dict:
    header = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, separator, value = line.partition(':')
        if not separator:
            continue
        header[key.strip().lower()] = value.strip()
    return header

@dataclass(frozen = True)
class Skill:
    name: str
    description: str
    body: str
    tools: tuple = ()
    path: str = ''

    def index_line(self) -> str:
        """One line for the system prompt: what it is, not how to do it."""
        return f'- {self.name}: {self.description}'

@dataclass
class Library:
    """The skills found on disk, and the ones that could not be used."""

    root: Path|None = None
    skills: dict = field(default_factory = dict)
    skipped: list = field(default_factory = list)

    def __bool__(self) -> bool:
        return bool(self.skills)

    def __len__(self) -> int:
        return len(self.skills)

    def names(self) -> list:
        return sorted(self.skills)

    def get(self, name: str) -> Skill|None:
        return self.skills.get((name or '').strip().lower())

    def index_text(self, limit: int = MAX_INDEX_CHARS) -> str:
        """The block the system prompt carries: names and descriptions only.

        Bounded, because a directory of skills would otherwise grow the prompt
        with every file added -- which is the cost this design exists to avoid.
        """
        if not self.skills:
            return ''
        lines = [skill.index_line() for _, skill in sorted(self.skills.items())]
        text = '\n'.join(lines)
        if len(text) > limit:
            text = text[:limit].rstrip() + f'\n... [index truncated at {limit} characters]'
        return text

def _skill_file(directory: Path) -> Path|None:
    if directory.is_dir():
        candidate = directory / SKILL_FILE
        if candidate.is_file():
            return candidate
        return None
    return directory if directory.is_file() else None

def read_skill(path: Path) -> tuple[Skill|None, str]:
    """(skill, reason) -- exactly one of them is set."""
    path = Path(path)
    try:
        text = path.read_text(errors = 'replace', encoding = 'utf-8')
    except OSError as e:
        return None, f'cannot read {path}: {type(e).__name__}'
    header, body = _split(text)
    name = (header.get('name') or (path.parent.name if path.name == SKILL_FILE else path.stem)).strip()
    if not name:
        return None, f'{path} has no usable name'
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        return None, f'{name!r} is not a valid skill name (letters, digits, underscore and dash)'
    description = (header.get('description') or '').strip()
    if not description:
        # The description is the only thing the model sees before loading, so a
        # skill without one is unusable rather than merely undocumented.
        return None, f'{name} has no description'
    tools = tuple(part.strip() for part in (header.get('tools') or '').split(',') if part.strip())
    return Skill(name = name, description = description, body = body, tools = tools,
                 path = str(path)), ''

def load(root: str|Path, enabled: bool = True) -> Library:
    """Every skill under a directory, sorted, with the unusable ones reported."""
    library = Library()
    if not enabled or not root:
        return library
    root = Path(root)
    library.root = root
    if not root.is_dir():
        return library
    candidates = sorted(path for path in root.iterdir() if not path.name.startswith('.'))
    for candidate in candidates:
        target = _skill_file(candidate)
        if target is None:
            continue
        skill, reason = read_skill(target)
        if skill is None:
            library.skipped.append(reason)
            continue
        if skill.name.lower() in library.skills:
            library.skipped.append(f'{skill.name} appears twice')
            continue
        library.skills[skill.name.lower()] = skill
    return library
