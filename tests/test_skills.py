"""Skills: written procedures the agent loads on demand.

A skill is a document. The description rides in the system prompt so the agent
knows it exists; the body costs nothing until the agent asks for it, and loading
one pins the tools it names without granting anything a tool call would not.
"""

import pytest

from mini_harness.agent import CORE_TOOLS, DeepSeekAgent
from mini_harness.config import Config
from mini_harness.selector import SELECTION
from mini_harness.skills import Skill, load, read_skill
from mini_harness.tool import box
from mini_harness.tool.box import TOOLS, SkillsInput, skills
from mini_harness.trace import TRACE
from tests.conftest import call, write

GOOD = """---
name: verify-a-change
description: Run the tests that cover an edit and report what they proved
tools: read_file, run_bash
---

1. Find the test file that covers the change.
2. Run it.
"""


def skill_dir(tmp_path, name='skills'):
    path = tmp_path / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def add(directory, folder, body=GOOD):
    write(directory / folder / 'SKILL.md', body)


# ------------------------------------------------------------------ parsing


def test_a_header_gives_a_name_a_description_and_tools(tmp_path):
    path = write(tmp_path / 'SKILL.md', GOOD)

    skill, reason = read_skill(path)

    assert reason == ''
    assert skill.name == 'verify-a-change'
    assert skill.tools == ('read_file', 'run_bash')
    assert skill.body.startswith('1. Find the test file')


def test_the_directory_name_is_the_fallback_name(tmp_path):
    add(tmp_path, 'review-a-patch', GOOD.replace('name: verify-a-change\n', ''))

    library = load(tmp_path)

    assert library.names() == ['review-a-patch']


def test_a_header_key_is_case_insensitive_and_spaces_do_not_matter(tmp_path):
    body = """---
Description:   Something useful
TOOLS: read_file ,   run_todo
---

Steps.
"""
    path = write(tmp_path / 'SKILL.md', body)

    skill, _reason = read_skill(path)

    assert skill.description == 'Something useful'
    assert skill.tools == ('read_file', 'run_todo')


def test_a_file_without_a_header_is_skipped_for_lacking_a_description(tmp_path):
    """The header is where the description lives, and it is not optional."""
    path = write(tmp_path / 'SKILL.md', 'Follow the deploy checklist.')

    skill, reason = read_skill(path)

    assert skill is None
    assert reason.endswith('has no description')


def test_a_skill_without_a_description_is_skipped_and_reported(tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    add(directory, 'bad', GOOD.replace(
        'description: Run the tests that cover an edit and report what they proved\n', ''))

    library = load(directory)

    assert library.names() == ['verify-a-change']
    assert any('has no description' in reason for reason in library.skipped)


def test_a_bad_name_is_rejected(tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'bad', GOOD.replace('name: verify-a-change', 'name: no spaces allowed'))

    library = load(directory)

    assert library.names() == []
    assert any('not a valid skill name' in reason for reason in library.skipped)


def test_two_skills_with_the_same_name_are_reported(tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'one')
    add(directory, 'two')

    library = load(directory)

    assert len(library) == 1
    assert any('appears twice' in reason for reason in library.skipped)


def test_a_missing_directory_is_an_empty_library_not_an_error(tmp_path):
    library = load(tmp_path / 'nowhere')

    assert library.names() == []
    assert library.skipped == []
    assert not library


def test_disabling_skills_loads_nothing(tmp_path):
    add(tmp_path, 'good')

    assert load(tmp_path, enabled=False).names() == []


def test_dotfiles_and_loose_files_are_ignored(tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    write(directory / 'notes.txt', 'not a skill')
    write(directory / '.hidden' / 'SKILL.md', GOOD)

    library = load(directory)

    assert library.names() == ['verify-a-change']


def test_a_skill_can_be_a_single_markdown_file(tmp_path):
    directory = skill_dir(tmp_path)
    write(directory / 'deploy.md', GOOD.replace('name: verify-a-change', 'name: deploy'))

    library = load(directory)

    assert library.names() == ['deploy']


# ------------------------------------------------------------------ the index


def test_the_index_carries_names_and_descriptions_only(tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')

    index = load(directory).index_text()

    assert 'verify-a-change' in index
    assert 'Run the tests that cover an edit' in index
    assert 'Run it.' not in index, 'the body must not ride in the prompt'


def test_the_index_is_bounded(tmp_path):
    directory = skill_dir(tmp_path)
    for index in range(40):
        add(directory, f'skill-{index}', GOOD.replace('verify-a-change', f'skill-{index}'))

    index = load(directory).index_text(limit=200)

    assert len(index) < 300
    assert 'index truncated at 200 characters' in index


def test_an_empty_library_indexes_to_nothing(tmp_path):
    assert load(tmp_path).index_text() == ''


# ------------------------------------------------------------------ the prompt


def test_the_system_prompt_carries_the_index(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))

    agent = DeepSeekAgent(TOOLS, cfg=cfg)

    assert '--- Skills ---' in agent.system[0]['content']
    assert 'verify-a-change' in agent.system[0]['content']
    assert 'Run it.' not in agent.system[0]['content']


def test_a_workspace_without_skills_adds_nothing(cfg):
    agent = DeepSeekAgent(TOOLS, cfg=cfg)

    assert agent.system[0]['content'] == cfg.system_prompt
    assert '--- Skills ---' not in agent.system[0]['content']


def test_skills_off_adds_nothing_even_when_the_directory_exists(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory), skills_enabled=False)

    agent = DeepSeekAgent(TOOLS, cfg=cfg)

    assert '--- Skills ---' not in agent.system[0]['content']


def test_the_skills_tool_is_never_hidden(cfg):
    """The prompt tells the agent to call it, so the budget must not take it away."""
    assert 'skills' in CORE_TOOLS


# ------------------------------------------------------------------ the tool


def listing(cfg):
    return skills(SkillsInput(action='list'), cfg=cfg)


def loading(cfg, name):
    return skills(SkillsInput(action='load', name=name), cfg=cfg)


def test_listing_shows_every_skill_and_its_description(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))

    report = listing(cfg)

    assert '1 available' in report
    assert 'verify-a-change: Run the tests that cover an edit' in report


def test_listing_reports_the_ones_it_could_not_use(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    add(directory, 'bad', 'no header, no description')
    cfg = cfg_factory(skills_dir=str(directory))

    report = listing(cfg)

    assert 'skipped:' in report
    assert 'description' in report


def test_listing_an_empty_directory_says_how_to_add_one(cfg):
    report = listing(cfg)

    assert 'none available' in report
    assert 'SKILL.md' in report


def test_loading_returns_the_procedure(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))

    report = loading(cfg, 'verify-a-change')

    assert 'verify-a-change -- Run the tests that cover an edit' in report
    assert '1. Find the test file that covers the change.' in report


def test_loading_pins_the_tools_the_skill_names(cfg_factory, tmp_path):
    """A procedure that needs a trimmed tool has to be able to get it back."""
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))
    SELECTION.register(TOOLS)

    loading(cfg, 'verify-a-change')

    assert {'read_file', 'run_bash'} <= SELECTION.active


def test_loading_names_a_tool_that_does_not_exist(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'stale',
        GOOD.replace('name: verify-a-change', 'name: stale')
            .replace('tools: read_file, run_bash', 'tools: read_file, teleport'))
    cfg = cfg_factory(skills_dir=str(directory))

    report = loading(cfg, 'stale')

    assert 'teleport' in report
    assert 'do not exist here' in report


def test_loading_an_unknown_skill_lists_what_is_available(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))

    report = loading(cfg, 'nope')

    assert "no skill named 'nope'" in report
    assert 'verify-a-change' in report


def test_loading_without_a_name_says_so(cfg):
    report = skills(SkillsInput(action='load'), cfg=cfg)

    assert 'needs a name' in report


def test_loading_is_case_insensitive_and_ignores_padding(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))

    assert '1. Find the test file' in loading(cfg, '  VERIFY-A-CHANGE ')


def test_loading_a_skill_with_no_body_still_answers(cfg_factory, tmp_path):
    directory = skill_dir(tmp_path)
    add(directory, 'empty', """---
name: empty
description: A skill that says nothing yet
---
""")
    cfg = cfg_factory(skills_dir=str(directory))

    assert 'no body' in loading(cfg, 'empty')


def test_loading_does_not_grant_permission(cfg_factory, tmp_path):
    """A skill is text: the risky tool it names still has to be approved."""
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))
    registry = {tool.name: tool for tool in TOOLS}
    execution = box.ToolExecution(registry, box._for_sub, cfg=cfg)

    result = execution.execute_tool(call('run_bash', command='ls'), cfg=cfg)

    assert not result.ok
    assert result.tag == 'denied'


# ------------------------------------------------------------------ tracing


def test_loading_a_skill_is_traced(cfg_factory, tmp_path, session_dir):
    import json
    directory = skill_dir(tmp_path)
    add(directory, 'good')
    cfg = cfg_factory(skills_dir=str(directory))
    path = session_dir / 'trace.jsonl'
    TRACE.configure(path)
    try:
        loading(cfg, 'verify-a-change')
    finally:
        TRACE.configure(None)

    event = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    load_event = [item for item in event if item['event'] == 'skill_load'][0]

    assert load_event['skill'] == 'verify-a-change'
    assert load_event['tools'] == ['read_file', 'run_bash']
    assert load_event['missing'] == []


# ------------------------------------------------------------------ configuration


def test_the_skills_directory_defaults_to_the_workspace(cfg_factory):
    cfg = cfg_factory()

    assert cfg.skills_path == cfg.work_space / 'skills'


def test_an_explicit_directory_wins(cfg_factory, tmp_path):
    cfg = cfg_factory(skills_dir=str(tmp_path))

    assert cfg.skills_path == tmp_path


def test_the_bench_profile_pins_skills_off():
    from mini_harness.bench_profile import BENCH_OVERRIDE

    assert BENCH_OVERRIDE['skills_enabled'] is False


def test_the_environment_can_override_both(monkeypatch, tmp_path):
    from mini_harness.config import build_config
    monkeypatch.setenv('MINI_HARNESS_SKILLS', 'false')
    monkeypatch.setenv('MINI_HARNESS_SKILLS_DIR', str(tmp_path))

    cfg = build_config()

    assert cfg.skills_enabled is False
    assert cfg.skills_dir == str(tmp_path)


def test_a_skill_is_comparable_and_immutable():
    skill = Skill(name='a', description='b', body='c', tools=('read_file',))
    with pytest.raises(Exception):
        skill.name = 'other'
    assert Config().skills_enabled is True
