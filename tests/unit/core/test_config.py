"""Core config: version hash, YAML loading, schema registry, prompt convention.

Uses a toy schema so core is tested without any use-case pack.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

from mdx_assessment.core.config import (
    ConfigError,
    ConfigSchemaRegistry,
    config_manifest,
    config_version,
    discover_prompts,
    load_model,
    load_prompt,
)


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor: float = Field(gt=0)
    name: str


class Playbook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str


def write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    write(root, "thresholds.yaml", "floor: 0.1\nname: rates\n")
    write(root, "playbooks/normal.yaml", "id: normal\n")
    write(root, "prompts/master_plan/v1.md", "Plan the run.\n")
    return root


@pytest.fixture
def registry() -> ConfigSchemaRegistry:
    reg = ConfigSchemaRegistry()
    reg.register("thresholds.yaml", Thresholds, owner="test")
    reg.register("playbooks/*.yaml", Playbook, owner="core")
    return reg


# --- version -------------------------------------------------------------------------


def test_version_is_deterministic_and_prefixed(cfg: Path) -> None:
    version = config_version(cfg)
    assert version.startswith("sha256:")
    assert len(version) == len("sha256:") + 64
    assert config_version(cfg) == version


def test_version_depends_on_content_not_location(cfg: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "elsewhere" / "config"
    shutil.copytree(cfg, copy)
    assert config_version(copy) == config_version(cfg)


@pytest.mark.parametrize(
    "change",
    ["edit", "add", "delete", "rename", "edit_prompt", "whitespace"],
)
def test_any_change_changes_the_version(cfg: Path, change: str) -> None:
    before = config_version(cfg)
    match change:
        case "edit":
            write(cfg, "thresholds.yaml", "floor: 0.2\nname: rates\n")
        case "add":
            write(cfg, "playbooks/high_volume.yaml", "id: high_volume\n")
        case "delete":
            (cfg / "playbooks/normal.yaml").unlink()
        case "rename":
            (cfg / "playbooks/normal.yaml").rename(cfg / "playbooks/normal_eod.yaml")
        case "edit_prompt":
            write(cfg, "prompts/master_plan/v1.md", "Plan the run carefully.\n")
        case "whitespace":
            write(cfg, "thresholds.yaml", "floor: 0.1\nname: rates\n\n")
    assert config_version(cfg) != before


def test_hidden_files_do_not_change_the_version(cfg: Path) -> None:
    before = config_version(cfg)
    write(cfg, ".DS_Store", "binary junk")
    write(cfg, "playbooks/.normal.yaml.swp", "editor swap")
    write(cfg, ".git/HEAD", "ref")
    assert config_version(cfg) == before
    assert ".DS_Store" not in config_manifest(cfg)


def test_missing_config_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        config_version(tmp_path / "nope")


# --- loader --------------------------------------------------------------------------


def test_load_model_ok(cfg: Path) -> None:
    assert load_model(cfg / "thresholds.yaml", Thresholds).floor == 0.1


def test_schema_error_names_file_and_field(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "floor: -1\nname: rates\n")
    with pytest.raises(ConfigError) as exc:
        load_model(cfg / "thresholds.yaml", Thresholds, display_name="thresholds.yaml")
    assert exc.value.problems == ["thresholds.yaml: floor: Input should be greater than 0"]


def test_all_schema_errors_are_reported(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "floor: -1\nsurprise: 1\n")
    with pytest.raises(ConfigError) as exc:
        load_model(cfg / "thresholds.yaml", Thresholds, display_name="t.yaml")
    assert sorted(exc.value.problems) == [
        "t.yaml: floor: Input should be greater than 0",
        "t.yaml: name: Field required",
        "t.yaml: surprise: Extra inputs are not permitted",
    ]


def test_duplicate_keys_are_rejected(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "floor: 0.1\nfloor: 0.5\nname: rates\n")
    with pytest.raises(ConfigError, match=r"duplicate key 'floor'.*line 2"):
        load_model(cfg / "thresholds.yaml", Thresholds)


def test_invalid_yaml_names_the_line(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "floor: 0.1\nname: [unclosed\n")
    with pytest.raises(ConfigError, match=r"thresholds.yaml: invalid YAML: .*line 3"):
        load_model(cfg / "thresholds.yaml", Thresholds, display_name="thresholds.yaml")


def test_empty_file_is_rejected(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "# only a comment\n")
    with pytest.raises(ConfigError, match="file is empty"):
        load_model(cfg / "thresholds.yaml", Thresholds)


def test_unsafe_yaml_tags_are_rejected(cfg: Path) -> None:
    write(cfg, "thresholds.yaml", "floor: !!python/object/apply:os.system ['echo hi']\nname: x\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_model(cfg / "thresholds.yaml", Thresholds)


# --- schema registry -----------------------------------------------------------------


def test_valid_tree_loads(cfg: Path, registry: ConfigSchemaRegistry) -> None:
    loaded = registry.load_config(cfg)
    assert loaded.version == config_version(cfg)
    assert loaded.get("thresholds.yaml", Thresholds).name == "rates"
    assert set(loaded.all_of(Playbook)) == {"playbooks/normal.yaml"}
    assert loaded.prompts == {"master_plan": ["v1"]}


def test_unowned_yaml_file_is_rejected(cfg: Path, registry: ConfigSchemaRegistry) -> None:
    write(cfg, "thresholdz.yaml", "floor: 1\nname: typo\n")
    with pytest.raises(ConfigError, match=r"thresholdz\.yaml: no registered schema owns this file"):
        registry.load_config(cfg)


def test_missing_required_file_is_reported(cfg: Path, registry: ConfigSchemaRegistry) -> None:
    (cfg / "thresholds.yaml").unlink()
    with pytest.raises(ConfigError, match=r"thresholds\.yaml: required by test but missing"):
        registry.load_config(cfg)


def test_problems_across_files_are_reported_together(
    cfg: Path, registry: ConfigSchemaRegistry
) -> None:
    write(cfg, "thresholds.yaml", "floor: 0\nname: rates\n")
    write(cfg, "playbooks/normal.yaml", "id: normal\nsteps: []\n")
    write(cfg, "prompts/Bad Name/v1.md", "x")
    with pytest.raises(ConfigError) as exc:
        registry.load_config(cfg)
    assert len(exc.value.problems) == 3
    assert "3 problem(s)" in str(exc.value)


def test_overlapping_schemas_are_rejected(cfg: Path, registry: ConfigSchemaRegistry) -> None:
    registry.register("playbooks/normal.*", Playbook, owner="other", required=False)
    with pytest.raises(ConfigError, match="claimed by more than one schema"):
        registry.load_config(cfg)


def test_same_pattern_cannot_be_registered_twice(registry: ConfigSchemaRegistry) -> None:
    with pytest.raises(ValueError, match="already registered"):
        registry.register("thresholds.yaml", Thresholds, owner="again")


def test_get_with_wrong_model_or_path(cfg: Path, registry: ConfigSchemaRegistry) -> None:
    loaded = registry.load_config(cfg)
    with pytest.raises(KeyError):
        loaded.get("thresholds.yaml", Playbook)
    with pytest.raises(KeyError):
        loaded.get("missing.yaml", Thresholds)


# --- prompts -------------------------------------------------------------------------


def test_prompt_versions_sorted_numerically(cfg: Path) -> None:
    for v in ("v2", "v10"):
        write(cfg, f"prompts/master_plan/{v}.md", f"version {v}\n")
    assert discover_prompts(cfg)["master_plan"] == ["v1", "v2", "v10"]
    assert load_prompt(cfg, "master_plan", "v10") == "version v10\n"


@pytest.mark.parametrize(
    ("rel", "message"),
    [
        ("prompts/MasterPlan/v1.md", "expected a folder named like"),
        ("prompts/master_plan/1.md", "expected a file named like 'v1.md'"),
        ("prompts/master_plan/v1.txt", "expected a file named like 'v1.md'"),
        ("prompts/notes.md", "expected a folder named like"),
    ],
)
def test_misnamed_prompts_are_rejected(cfg: Path, rel: str, message: str) -> None:
    write(cfg, rel, "text\n")
    with pytest.raises(ConfigError, match=message):
        discover_prompts(cfg)


def test_empty_prompt_is_rejected(cfg: Path) -> None:
    write(cfg, "prompts/master_review/v1.md", "  \n")
    with pytest.raises(ConfigError, match="prompt is empty"):
        discover_prompts(cfg)


def test_unknown_prompt(cfg: Path) -> None:
    with pytest.raises(ConfigError, match="prompt not found"):
        load_prompt(cfg, "master_plan", "v9")
    with pytest.raises(ConfigError, match="prompt not found"):
        load_prompt(cfg, "../secrets", "v1")
