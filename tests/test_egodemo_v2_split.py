import json
from pathlib import Path

import pytest

from src.dataset.egodemo_v2_split import build_groups, freeze_split


def record(uid, scene=None, environment=None, **kwargs):
    return dict(source_episode_uuid=uid, scene_id=scene, environment_id=environment,
                task_name="task", manifest_frame_count=80, planned_assets=[], **kwargs)


def test_transitive_scene_environment_and_content_aliases():
    rows = [record("a", "s1", "e1"), record("b", "s2", "e1"),
            record("c", "s2", "e2"), record("d", "s3", "e3"), record("e")]
    rows[2]["planned_assets"] = [dict(role="data", expected_sha256="shared")]
    rows[3]["planned_assets"] = [dict(role="data", expected_sha256="shared")]
    assert sorted(g["sequence_ids"] for g in build_groups(rows)) == [["a", "b", "c", "d"], ["e"]]


def test_unknown_subjects_do_not_merge_and_verified_subjects_do():
    rows = [record("a", subject_id="unknown"), record("b", subject_id="unknown")]
    assert len(build_groups(rows, True)) == 2
    for row in rows:
        row["subject_id"] = "p1"
    assert len(build_groups(rows)) == 2
    assert len(build_groups(rows, True)) == 1


def source(tmp_path):
    rows = [record(str(i), "scene" + str(i), "env" + str(i)) for i in range(20)]
    rows[0]["previously_downloaded"] = True
    rows[1]["environment_id"] = rows[0]["environment_id"]
    path = tmp_path / "source" / "selection_frozen.json"
    path.parent.mkdir()
    path.write_text(json.dumps(dict(records=rows)))
    for row in rows:
        staging = path.parent / "staging" / row["source_episode_uuid"]
        staging.mkdir(parents=True)
        (staging / "sequence.npz").write_bytes(row["source_episode_uuid"].encode())
        (staging / "meta.json").write_text("{}")
    return path


def test_seen_test_and_linked_group_are_development_only_and_freeze_immutable(tmp_path):
    selection = source(tmp_path)
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps(dict(test=["2"])))
    root = tmp_path / "out"
    kwargs = dict(prior_splits_paths=[prior])
    split = freeze_split(selection, root, **kwargs)
    assert {split["assignments"][uid] for uid in ["0", "1", "2"]} == {"train"}
    assert split["counts"] == dict(train=14, val=3, test=3)
    assert freeze_split(selection, root, **kwargs) == split
    assert freeze_split(selection, tmp_path / "out2", **kwargs) == split
    with pytest.raises(ValueError, match="immutable"):
        freeze_split(selection, root, seed=2, **kwargs)
    assert split["subject_disjoint_verified"] is False
    assert split["training_allowed"] is False


def test_source_mutation_refuses_to_reuse_split(tmp_path):
    selection = source(tmp_path)
    root = tmp_path / "out"
    freeze_split(selection, root)
    (selection.parent / "staging/0/sequence.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable"):
        freeze_split(selection, root)


def test_duplicate_uuid_rejected(tmp_path):
    selection = source(tmp_path)
    data = json.loads(selection.read_text())
    data["records"].append(data["records"][0])
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        freeze_split(selection, tmp_path / "out")
