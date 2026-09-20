"""The tool contract is enforced in code, not just described to the model.

Every rule here used to live only in the system prompt or in the shape of the
built-in tools. A definition that breaks one of them fails at dispatch time,
which is both late and silent, so the checks run at definition time instead.
"""

import pytest
from pydantic import BaseModel, ConfigDict

from mini_harness.tool import box
from mini_harness.tool.box import ToolDefinition
from tests.conftest import REGISTRY, call


class PlainInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str


class PathInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    file_path: str


def make(**overrides):
    """A minimal valid definition with any field replaceable."""
    fields = {
        'name': 'sample',
        'description': 'a sample tool',
        'parameters': PlainInput,
        'function': lambda args, cfg=None: 'ok',
        'risky': False,
    }
    fields.update(overrides)
    return ToolDefinition(**fields)


# --------------------------------------------------------------------------- signatures


def test_a_function_accepting_cfg_is_recorded():
    assert make().wants_cfg is True


def test_a_function_without_cfg_is_still_supported():
    assert make(function=lambda args: 'ok').wants_cfg is False


def test_var_keyword_counts_as_accepting_cfg():
    def tool(args, **kwargs):
        return 'ok'

    assert make(function=tool).wants_cfg is True


def test_keyword_only_cfg_is_accepted():
    def tool(args, *, cfg=None):
        return 'ok'

    assert make(function=tool).wants_cfg is True


def test_positional_only_cfg_is_rejected():
    def tool(args, cfg, /):
        return 'ok'

    with pytest.raises(ValueError, match='positional-only'):
        make(function=tool)


def test_a_function_taking_no_arguments_is_rejected():
    with pytest.raises(ValueError, match='first parameter'):
        make(function=lambda: 'ok')


def test_a_non_callable_function_is_rejected():
    with pytest.raises(ValueError, match='callable'):
        make(function='not callable')


# --------------------------------------------------------------------------- fields


@pytest.mark.parametrize('name', ['', 'has space', 'has.dot', 'x' * 65, None, 7])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValueError, match='valid tool name'):
        make(name=name)


def test_a_missing_description_is_rejected():
    with pytest.raises(ValueError, match='description'):
        make(description='   ')


def test_parameters_must_be_a_basemodel_subclass():
    with pytest.raises(ValueError, match='BaseModel'):
        make(parameters=dict)


def test_risky_must_be_a_bool():
    with pytest.raises(ValueError, match='bool'):
        make(risky='yes')


# --------------------------------------------------------------------------- registry


def test_the_built_in_registry_satisfies_its_own_contract():
    box.validate_tools(box.TOOLS)


def test_every_built_in_tool_accepts_cfg():
    assert all(tool.wants_cfg for tool in box.TOOLS)


def test_duplicate_names_are_rejected():
    with pytest.raises(RuntimeError, match='duplicate'):
        box.validate_tools([make(), make()])


def test_a_missing_core_tool_is_reported_as_drift():
    with pytest.raises(RuntimeError, match='drift'):
        box.validate_tools([make()])


def core_registry(**overrides):
    """A registry holding every required core tool, with fields replaceable."""
    base = {
        'read_file': PathInput,
        'grep_file': PlainInput,
        'write_file': PathInput,
        'edit_file': PathInput,
    }
    base.update(overrides)
    return [make(name=name, parameters=parameters) for name, parameters in base.items()]


@pytest.mark.parametrize('name', ['write_file', 'edit_file', 'read_file'])
def test_path_keyed_tools_must_declare_file_path(name):
    """Without file_path the gate would silently skip every check."""
    with pytest.raises(RuntimeError, match='file_path'):
        box.validate_tools(core_registry(**{name: PlainInput}))


def test_grep_file_is_not_path_keyed():
    """grep records reads from its output, not from an argument."""
    assert 'grep_file' in box.READ_TOOLS
    assert 'grep_file' not in box.PATH_KEYED_TOOLS
    assert 'path' in box.GrepFileInput.model_fields


# --------------------------------------------------------------------------- dispatch


def test_the_executor_supports_both_calling_conventions(cfg):
    seen = []

    def with_cfg(args, cfg=None):
        seen.append(('with_cfg', cfg is not None))
        return 'a'

    def without_cfg(args):
        seen.append(('without_cfg', True))
        return 'b'

    registry = dict(REGISTRY)
    registry['with_cfg'] = make(name='with_cfg', function=with_cfg)
    registry['without_cfg'] = make(name='without_cfg', function=without_cfg)
    runner = box.ToolExecution(registry, box._always_allow, cfg=cfg)

    assert runner.execute_tool(call('with_cfg', text='x'), cfg=cfg).content == 'a'
    assert runner.execute_tool(call('without_cfg', text='x'), cfg=cfg).content == 'b'
    assert seen == [('with_cfg', True), ('without_cfg', True)]


def test_wants_cfg_is_not_settable_by_callers():
    definition = make()
    with pytest.raises(Exception):
        definition.wants_cfg = False
