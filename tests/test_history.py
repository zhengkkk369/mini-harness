"""Repairing a stored conversation into one the API will accept.

A ``tool_calls`` entry with no matching ``tool`` message makes the next request
fail outright, so this is the one place where a saved session must be fixed
rather than merely loaded.
"""

from mini_harness.history import PLACEHOLDER, paired_history, unanswered


def call(call_id, name='read_file'):
    return {'id': call_id, 'type': 'function',
            'function': {'name': name, 'arguments': '{}'}}


def assistant(*call_ids):
    return {'role': 'assistant', 'content': '',
            'tool_calls': [call(call_id) for call_id in call_ids]}


def result(call_id, content='ok'):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': content}


def test_an_empty_history_stays_empty():
    assert paired_history([]) == []


def test_a_complete_pair_is_left_alone():
    messages = [{'role': 'user', 'content': 'hi'}, assistant('c1'), result('c1'),
                {'role': 'assistant', 'content': 'done'}]
    assert paired_history(messages) == messages


def test_a_plain_conversation_is_untouched():
    messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                {'role': 'assistant', 'content': 'a'}]
    assert paired_history(messages) == messages


def test_a_missing_result_is_filled_in_place():
    messages = [{'role': 'user', 'content': 'go'}, assistant('c1'),
                {'role': 'assistant', 'content': 'I stopped'}]

    repaired = paired_history(messages)

    assert [m['role'] for m in repaired] == ['user', 'assistant', 'tool', 'assistant']
    filler = repaired[2]
    assert filler['tool_call_id'] == 'c1'
    assert filler['content'] == PLACEHOLDER


def test_a_trailing_missing_result_is_appended():
    messages = [{'role': 'user', 'content': 'go'}, assistant('c1')]

    repaired = paired_history(messages)

    assert repaired[-1] == {'role': 'tool', 'tool_call_id': 'c1', 'content': PLACEHOLDER}


def test_only_the_unanswered_call_is_filled():
    messages = [assistant('c1', 'c2'), result('c1')]

    repaired = paired_history(messages)

    assert [m['role'] for m in repaired] == ['assistant', 'tool', 'tool']
    assert repaired[2]['tool_call_id'] == 'c2'


def test_several_missing_results_are_all_filled():
    messages = [assistant('c1', 'c2', 'c3')]

    repaired = paired_history(messages)

    assert [m['tool_call_id'] for m in repaired[1:]] == ['c1', 'c2', 'c3']


def test_fillers_go_before_the_next_message_not_after_it():
    """The API wants each result immediately after the call that owes it."""
    messages = [assistant('c1'), {'role': 'user', 'content': 'are you there'}]

    repaired = paired_history(messages)

    assert [m['role'] for m in repaired] == ['assistant', 'tool', 'user']


def test_an_orphan_tool_result_is_left_alone():
    """It is not ours to delete; the repair only ever adds."""
    messages = [{'role': 'user', 'content': 'u'}, result('ghost')]
    assert paired_history(messages) == messages


def test_an_assistant_without_tool_calls_owes_nothing():
    messages = [{'role': 'assistant', 'content': 'just talking'}]
    assert paired_history(messages) == messages


def test_the_input_list_is_not_modified():
    messages = [assistant('c1')]
    before = list(messages)

    paired_history(messages)

    assert messages == before
    assert len(messages) == 1


def test_a_call_with_no_id_does_not_break_the_repair():
    """Older sessions may hold a malformed call; resuming must still work."""
    messages = [{'role': 'assistant', 'content': '',
                 'tool_calls': [{'type': 'function', 'function': {'name': 'x'}}]}]
    repaired = paired_history(messages)
    assert len(repaired) == 1


def test_unanswered_reports_the_outstanding_ids():
    messages = [assistant('c1', 'c2'), result('c1')]
    assert unanswered(messages) == ['c2']


def test_unanswered_is_empty_for_a_healthy_history():
    assert unanswered([assistant('c1'), result('c1')]) == []
