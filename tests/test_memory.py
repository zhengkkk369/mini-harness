"""Retrievable memory: the messages compaction removed stay searchable."""

import json

import pytest

from mini_harness.memory import JOURNAL_NAME, Memory, journal_for, journal_path
from mini_harness.tool import box
from mini_harness.trace import TRACE
from tests.conftest import call, write


def journal(path, *records):
    """Write journal lines the way compaction does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


def message(role, content='', tool_calls=None):
    entry = {'role': role, 'content': content}
    if tool_calls:
        entry['tool_calls'] = tool_calls
    return entry


def tool_call(name, arguments):
    return {'function': {'name': name, 'arguments': arguments}}


# --------------------------------------------------------------------------- journal path


def test_the_journal_sits_next_to_the_session_file(cfg_factory, session_dir):
    cfg = cfg_factory()
    assert journal_path(cfg) == session_dir / JOURNAL_NAME
    assert journal_for(session_dir / 'session.json') == session_dir / JOURNAL_NAME


def test_without_a_session_path_the_journal_uses_the_workspace(cfg_factory):
    cfg = cfg_factory(session_path=None)
    assert journal_path(cfg) == cfg.work_space / JOURNAL_NAME


def test_the_writer_and_the_reader_agree(cfg_factory, session_dir):
    """compact.py writes through journal_for; the tool reads through journal_path."""
    cfg = cfg_factory()
    assert journal_for(cfg.session_path, cfg=cfg) == journal_path(cfg)


# --------------------------------------------------------------------------- indexing


def test_an_absent_journal_is_empty(cfg, session_dir):
    memory = Memory(session_dir / JOURNAL_NAME)
    assert memory.entries() == []
    assert memory.search('anything') == []
    assert memory.stats() == {'entries': 0, 'bytes': 0, 'roles': {}}


def test_entries_are_flattened_from_the_removed_list(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 100.0,
        'removed': [
            message('user', 'the deploy window is 02:00 to 04:00'),
            message('assistant', 'noted'),
            message('assistant', '', [tool_call('read_file', '{"file_path": "a.py"}')]),
            message('tool', 'file contents here'),
            message('user', '   '),
        ],
    })

    entries = Memory(path).entries()

    assert [e.role for e in entries] == ['user', 'assistant', 'assistant', 'tool']
    assert entries[0].ts == 100.0
    assert 'read_file' in entries[2].text
    assert entries[0].index == 0


def test_malformed_lines_and_records_are_skipped(cfg, session_dir):
    path = session_dir / JOURNAL_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        'not json\n'
        '\n'
        '{"ts": 1, "removed": "not a list"}\n'
        '{"ts": 2, "removed": [{"role": "user", "content": "kept"}]}\n'
        '{"ts": 3}\n',
        encoding='utf-8')

    entries = Memory(path).entries()

    assert [e.text for e in entries] == ['kept']


def test_the_index_is_rebuilt_when_the_journal_grows(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {'ts': 1, 'removed': [message('user', 'first')]})
    memory = Memory(path)
    assert len(memory.entries()) == 1

    journal(path, {'ts': 2, 'removed': [message('user', 'second')]})

    assert [e.text for e in memory.entries()] == ['first', 'second']


# --------------------------------------------------------------------------- search


def test_search_ranks_the_matching_entry_first(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [
            message('user', 'the deploy window is 02:00 to 04:00'),
            message('user', 'the on-call rotation is weekly'),
            message('tool', 'unrelated output about deploy scripts'),
        ],
    })

    hits = Memory(path).search('deploy window')

    assert hits
    assert 'deploy window' in hits[0].text
    assert hits[0].score > hits[-1].score


def test_a_rare_term_decides_the_ranking(cfg, session_dir):
    """A term present in one entry outweighs one present in all of them."""
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [message('tool', 'alpha beta'),
                    message('tool', 'alpha gamma'),
                    message('tool', 'alpha delta')],
    })

    hits = Memory(path).search('alpha beta')

    assert hits[0].text == 'alpha beta'
    # 'alpha' alone still matches everything, so all three come back
    assert len(hits) == 3


def test_search_honours_the_limit(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [message('user', f'alpha item {i}') for i in range(10)],
    })

    assert len(Memory(path).search('alpha', limit=3)) == 3


def test_search_can_filter_by_role(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [message('user', 'alpha from the user'),
                    message('tool', 'alpha from a tool')],
    })
    memory = Memory(path)

    assert [h.role for h in memory.search('alpha', role='tool')] == ['tool']
    assert [h.role for h in memory.search('alpha', role='user')] == ['user']
    assert len(memory.search('alpha')) == 2


def test_search_is_deterministic_for_equal_scores(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [message('user', 'alpha one'), message('user', 'alpha two')],
    })
    memory = Memory(path)

    assert [h.text for h in memory.search('alpha')] == [h.text for h in memory.search('alpha')]


def test_a_query_with_no_usable_terms_matches_nothing(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {'ts': 1, 'removed': [message('user', 'alpha')]})
    assert Memory(path).search('   ...   ') == []


def test_a_term_in_no_entry_matches_nothing(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {'ts': 1, 'removed': [message('user', 'alpha')]})
    assert Memory(path).search('zzzz') == []


def test_stats_report_the_role_mix(cfg, session_dir):
    path = journal(session_dir / JOURNAL_NAME, {
        'ts': 1,
        'removed': [message('user', 'a'), message('tool', 'b'), message('tool', 'c')],
    })

    stats = Memory(path).stats()

    assert stats['entries'] == 3
    assert stats['roles'] == {'user': 1, 'tool': 2}
    assert stats['bytes'] > 0


# --------------------------------------------------------------------------- the tool


def test_recall_reports_an_empty_archive(cfg):
    result = box.recall(box.RecallInput(query='anything'), cfg=cfg)
    assert 'nothing has been compacted yet' in result


def test_recall_returns_the_matching_text(cfg_factory, session_dir):
    cfg = cfg_factory()
    journal(journal_path(cfg), {'ts': 1, 'removed': [
        message('user', 'the deploy window is 02:00 to 04:00 UTC'),
        message('tool', 'unrelated'),
    ]})

    result = box.recall(box.RecallInput(query='deploy window'), cfg=cfg)

    assert '02:00 to 04:00' in result
    assert '1 of 2 archived messages' in result
    assert 'score' in result


def test_recall_reports_a_miss_without_inventing_anything(cfg_factory):
    cfg = cfg_factory()
    journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', 'alpha')]})

    result = box.recall(box.RecallInput(query='zzzz'), cfg=cfg)

    assert 'no match' in result
    assert 'alpha' not in result


def test_recall_truncates_long_matches(cfg_factory):
    cfg = cfg_factory(recall_snippet=20)
    journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', 'needle ' + 'x' * 200)]})

    result = box.recall(box.RecallInput(query='needle'), cfg=cfg)

    assert 'more chars]' in result
    assert 'x' * 200 not in result


def test_recall_can_be_switched_off(cfg_factory):
    cfg = cfg_factory(recall_enabled=False)
    journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', 'alpha')]})

    assert box.recall(box.RecallInput(query='alpha'), cfg=cfg) == '[recall]: disabled by configuration'


def test_recall_rejects_a_bad_limit(cfg):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        box.RecallInput(query='x', limit=0)
    with pytest.raises(ValidationError):
        box.RecallInput(query='x', limit=99)


def test_recall_is_traced(cfg_factory, session_dir):
    cfg = cfg_factory()
    journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', 'alpha')]})
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        box.recall(box.RecallInput(query='alpha'), cfg=cfg)
    finally:
        TRACE.configure(None)

    event = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()][0]
    assert event['event'] == 'recall'
    assert event['hits'] == 1
    assert event['entries'] == 1


def test_recall_is_dispatched_like_any_other_tool(cfg_factory):
    """It goes through the executor, so tags, tracing and policy all apply."""
    from tests.conftest import executor
    cfg = cfg_factory()
    journal(journal_path(cfg), {'ts': 1, 'removed': [message('user', 'alpha detail')]})

    result = executor(cfg).execute_tool(call('recall', query='alpha'), cfg=cfg)

    assert result.ok is True
    assert 'alpha detail' in result.content


def test_recall_is_side_effect_free(cfg):
    assert 'recall' in box.SIDE_EFFECT_FREE_TOOLS
