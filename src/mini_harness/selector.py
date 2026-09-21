"""Decide which tools to put in front of the model.

Every tool schema is sent on every request, so a large tool surface is a fixed
cost per turn. With one MCP server attached the schemas were about a quarter of
the first request's tokens, and that grows with every server.

When the surface is bigger than a budget, this ranks tools against the task and
exposes the best matches. Two things are never hidden: the core tools the
locate/understand workflow starts from, and anything the model has pulled in
through `find_tools`. That escape hatch is what makes hiding safe -- without it,
a tool the ranking did not think of would be unreachable. A call to a hidden
tool is answered with an instruction to use `find_tools` rather than executed,
so guessing a name gets the model nowhere.

Off by default (``tool_budget = 0`` means "expose everything"): the ranking is
lexical, and a task whose wording does not resemble a tool description is
exactly where a budget would hurt.
"""

from mini_harness.memory import rank, tokens

class Selection:
    """The full catalogue for a run, and the tools asked for by name."""

    def __init__(self) -> None:
        self.catalog: list = []
        self.active: set = set()
        return

    def register(self, definitions: list) -> None:
        self.catalog = list(definitions)
        self.active = set()
        return

    def activate(self, names) -> None:
        self.active.update(names)
        return

    def reset(self) -> None:
        self.catalog, self.active = [], set()
        return

SELECTION = Selection()

def describe(definition) -> str:
    """What a tool is matched against: its name and its own description."""
    return f'{definition.name} {definition.description}'

def rank_definitions(query: str, definitions: list) -> list:
    """Definitions best matching the query, then the rest in their given order."""
    if not definitions:
        return []
    tokenised = [tuple(tokens(describe(definition))) for definition in definitions]
    ordered = [index for index, _ in rank(set(tokens(query)), tokenised)]
    ordered += [index for index in range(len(definitions)) if index not in ordered]
    return [definitions[index] for index in ordered]

def select(task: str, definitions: list, always = (), budget: int = 0) -> list:
    """The definitions to expose for this task.

    A budget of zero, or a tool set that already fits inside it, exposes
    everything. Otherwise the pinned tools are kept and the rest are ranked
    against the task.
    """
    if budget <= 0 or len(definitions) <= budget:
        return list(definitions)

    pinned_names = set(always) | SELECTION.active
    pinned = [d for d in definitions if d.name in pinned_names]
    rest = [d for d in definitions if d.name not in pinned_names]
    room = budget - len(pinned)
    if room <= 0:
        return pinned
    return pinned + rank_definitions(task, rest)[:room]
