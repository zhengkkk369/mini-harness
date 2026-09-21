"""What counts as having checked a change.

The rule used to be "a shell tool ran": `echo ok` satisfied it. It is now "a run
touched what changed, or ran the suite", which is a heuristic and is tested as
one -- the point is to catch a run that checked nothing, not to prove that a
check was good.
"""

import pytest

from mini_harness.agent import (VERIFY_RUNNERS, normalise_path, verification_touches,
                                verify_nudge)
from mini_harness.agent import Result
from mini_harness.config import build_config


CHANGED = {'sandbox/parser.py'}


# ------------------------------------------------------------------ the verdict


def test_running_the_changed_file_is_targeted():
    assert verification_touches('python sandbox/parser.py', '', CHANGED) == 'targeted'


def test_the_file_name_alone_is_enough():
    assert verification_touches('python parser.py', '', CHANGED) == 'targeted'


def test_a_test_file_named_after_the_change_is_targeted():
    """`pytest tests/test_parser.py` is how a change is normally checked."""
    assert verification_touches('python -m pytest tests/test_parser.py', '', CHANGED) == 'targeted'


def test_the_output_counts_too():
    """A traceback naming the file is evidence the change was exercised."""
    assert verification_touches('python -m pytest', 'FAILED tests/parser_test: parser.py:12',
                                CHANGED) == 'targeted'


def test_running_the_suite_counts_on_its_own():
    for command in ('pytest', 'uv run --locked pytest', 'python -m unittest discover',
                    'npm test', 'cargo test'):
        assert verification_touches(command, '', CHANGED) == 'suite', command


def test_running_something_unrelated_does_not_count():
    for command in ('echo ok', 'ls -la', 'git status', 'sleep 1', 'cat README.md'):
        assert verification_touches(command, '', CHANGED) == 'unrelated', command


def test_a_short_stem_does_not_match_by_accident():
    """A file called f.py must not be "covered" by any word containing f."""
    assert verification_touches('echo for the win', '', {'sandbox/f.py'}) == 'unrelated'
    assert verification_touches('python sandbox/f.py', '', {'sandbox/f.py'}) == 'targeted'


def test_windows_separators_match_either_way():
    assert verification_touches('python sandbox\\parser.py', '', CHANGED) == 'targeted'
    assert verification_touches('python sandbox/parser.py', '', {'sandbox\\parser.py'}) == 'targeted'


def test_matching_is_case_insensitive():
    assert verification_touches('python SANDBOX/Parser.py', '', CHANGED) == 'targeted'


def test_nothing_changed_means_nothing_to_cover():
    assert verification_touches('echo ok', '', set()) == 'unrelated'
    assert verification_touches('pytest', '', set()) == 'suite'


def test_the_runner_list_is_deliberately_short():
    assert 'pytest' in VERIFY_RUNNERS
    assert len(VERIFY_RUNNERS) <= 20


# ------------------------------------------------------------------ the nudge


def test_the_nudge_names_the_files_when_it_knows_them():
    text = verify_nudge({'sandbox/parser.py'})

    assert 'sandbox/parser.py' in text
    # The experiment and the tests key on this phrase staying stable.
    assert 'have not run anything since' in text


def test_the_nudge_stays_generic_without_files():
    text = verify_nudge(set())

    assert text.startswith('You changed files but')
    assert 'have not run anything since' in text


def test_the_nudge_summarises_a_long_list():
    text = verify_nudge({f'sandbox/f{index}.py' for index in range(7)})

    assert '(and 3 more)' in text


def test_paths_are_normalised_for_display():
    assert normalise_path('  SANDBOX\\Parser.py ') == 'sandbox/parser.py'


# ------------------------------------------------------------------ reporting


def test_the_result_fields_default_so_older_calls_still_build():
    result = Result('completed', 1, 1, 1, {}, {}, 5, 5, 5, 0.5)

    assert result.verification == 'none'
    assert result.unverified_files == ()


def test_the_environment_can_relax_the_rule(monkeypatch):
    monkeypatch.setenv('MINI_HARNESS_VERIFY_TARGETS', 'false')

    assert build_config().verify_targets_changed is False


def test_the_bench_profile_pins_the_rule_to_the_recorded_behaviour():
    from mini_harness.bench_profile import BENCH_OVERRIDE

    assert BENCH_OVERRIDE['verify_targets_changed'] is False
