"""Dispatch policy: which tool calls are refused before they run.

The policy is deliberately separate from approval. Approval asks a human about
a risky call; the policy is the part that does not need to ask, because the
answer is already known: this tool is not allowed here, or this command matches
a pattern that must never run unattended.

Deny patterns are matched against the command when the tool takes one (so
`curl * | sh` reads naturally) and against the raw arguments otherwise.
"""

import fnmatch
import json

from dataclasses import dataclass, field

ALLOW = 'allow'
DENY = 'deny'

@dataclass(frozen = True)
class Decision:
    decision: str = ALLOW
    reason: str = ''

    @property
    def denied(self) -> bool:
        return self.decision == DENY

@dataclass
class Policy:
    deny_tools: frozenset = frozenset()
    deny_patterns: tuple = ()
    read_only: bool = False
    mutating_tools: frozenset = frozenset()

    def subject(self, tool_call) -> str:
        """What a deny pattern is matched against."""
        raw = getattr(tool_call.function, 'arguments', '') or ''
        try:
            args = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if isinstance(args, dict) and isinstance(args.get('command'), str):
            return args['command']
        return raw

    def decide(self, tool_call) -> Decision:
        name = tool_call.function.name
        if name in self.deny_tools:
            return Decision(DENY, f'{name} is denied by policy')
        if self.read_only and name in self.mutating_tools:
            return Decision(DENY, f'{name} is denied: this run is read-only')
        subject = self.subject(tool_call)
        for pattern in self.deny_patterns:
            if fnmatch.fnmatch(subject, pattern):
                return Decision(DENY, f'{name} is denied by pattern {pattern!r}')
        return Decision(ALLOW)

    @classmethod
    def from_config(cls, cfg, mutating_tools = frozenset()) -> 'Policy':
        return cls(
            deny_tools = frozenset(cfg.policy_deny_tools),
            deny_patterns = tuple(cfg.policy_deny_patterns),
            read_only = cfg.read_only,
            mutating_tools = frozenset(mutating_tools),
        )
