"""Repair a stored conversation into one an API will accept.

The chat API requires every ``tool_calls`` entry on an assistant message to be
followed by a ``tool`` message carrying its id. A session saved mid-turn, or
cancelled between the two, breaks that pairing, and the next request is rejected
outright rather than degraded.

`paired_history` fills the gaps with an explicit placeholder so the conversation
can be resumed. It lives in the package rather than in the TUI because the
worker, the session loader and any future front end all need the same repair.
"""

PLACEHOLDER = '[interrupted] No result received. Check current files before retrying.'

def paired_history(messages: list) -> list:
    """Return a copy with a placeholder result for every unanswered tool call.

    Messages themselves are not copied or altered; only missing results are
    added, in the position the API expects them.
    """
    repaired, pending = [], {}
    for message in messages:
        if message.get('role') != 'tool' and pending:
            repaired.extend({'role': 'tool', 'tool_call_id': key, 'content': PLACEHOLDER}
                            for key in pending)
            pending = {}
        repaired.append(message)
        if message.get('role') == 'assistant':
            # A malformed call carries no id and can never be answered, so it
            # must be skipped rather than crashing the repair it needs.
            pending = {call.get('id'): True for call in message.get('tool_calls') or []
                       if isinstance(call, dict) and call.get('id')}
        elif message.get('role') == 'tool':
            pending.pop(message.get('tool_call_id'), None)
    repaired.extend({'role': 'tool', 'tool_call_id': key, 'content': PLACEHOLDER}
                    for key in pending)
    return repaired


def unanswered(messages: list) -> list:
    """The tool_call ids the conversation still owes a result for."""
    return [message['tool_call_id'] for message in paired_history(messages)
            if message.get('role') == 'tool' and message.get('content') == PLACEHOLDER]
