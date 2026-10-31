import copy
import json
import numpy as np
import pytest
from dual_arm_calibration.demo import generate
from dual_arm_calibration.workflow import solve, digest, table_frame, check_rig
from dual_arm_calibration.geometry import transform_error
from dual_arm_calibration.storage import save_run, publish
from dual_arm_calibration.validation import transform, camera_model
from dual_arm_calibration.wrist import solve_wrist
from dual_arm_calibration.board import Board
from dual_arm_calibration.intrinsics import solve as fit_lens


@pytest.fixture(scope="module")
def solved():
    rig, data, truth = generate()
    return rig, data, truth, solve(rig, data)


def test_six_and_seven_joint_rig_without_table_or_intrinsic_fit(solved):
    rig, data, truth, result = solved
    assert result["quality_passed"]
    assert result["table"] == {"calibrated": False, "table_z_m": None}
    assert result["world_frame"] == "top_camera"
    np.testing.assert_allclose(result["world_from_top"], np.eye(4))
    for arm in ("left", "right"):
        mm, deg = transform_error(
            result["wrists"][arm]["flange_from_camera"], truth["wrists"][arm]
        )
        assert mm < 0.01 and deg < 0.01
        mm, deg = transform_error(
            result["bridges"][arm]["base_from_top"], truth["bridges"][arm]
        )
        assert mm < 0.01 and deg < 0.01
    assert len(data["wrists"]["left"][0]["joint_angles_rad"]) == 6
    assert len(data["wrists"]["right"][0]["joint_angles_rad"]) == 7


def test_optional_table_and_thickness():
    rig, data, truth = generate(table=True)
    result = solve(rig, data)
    assert result["world_frame"] == "table"
    assert result["table"]["table_z_m"] == 0
    board = np.array(data["table"][0]["cameras"]["top"]["camera_from_target"])
    np.testing.assert_allclose(
        np.array(result["world_from_top"]) @ board,
        result["table"]["world_from_board"],
        atol=1e-6,
    )
    assert result["table"]["world_from_board"][2][3] == 0.008


def test_binding_changes_are_rejected(solved):
    rig, data, _, _ = solved
    rig = copy.deepcopy(rig)
    rig["arms"]["right"]["model_id"] = "reinstalled"
    with pytest.raises(ValueError, match="bound"):
        solve(rig, data)


def test_cross_batch_reuse_is_rejected(solved):
    rig, data, _, _ = solved
    data = copy.deepcopy(data)
    data["validation"][0]["id"] = data["wrists"]["left"][0]["id"]
    with pytest.raises(ValueError, match="overlap"):
        solve(rig, data)


def test_degenerate_motion_is_rejected(solved):
    rig, data, _, _ = solved
    samples = copy.deepcopy(data["wrists"]["left"])
    for s in samples:
        s["base_from_flange"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="coverage"):
        solve_wrist(
            samples,
            Board(rig["board"]).object_points,
            camera_model=rig["cameras"]["left"]["model"],
        )


def test_validation_failure_prevents_publication(solved):
    rig, data, _, _ = solved
    data = copy.deepcopy(data)
    data["validation"][0]["base_from_flange"]["left"][0][3] += 0.04
    with pytest.raises(ValueError, match="cross-camera"):
        solve(rig, data)


def test_d435i_fit_profile_cannot_be_used_for_other_cameras():
    with pytest.raises(ValueError, match="D435i RGB"):
        fit_lens([], [], camera_family="another-camera", stream="rgb")
    with pytest.raises(ValueError, match="D435i RGB"):
        fit_lens([], [], camera_family="realsense_d435i", stream="depth")


@pytest.mark.parametrize(
    "value", [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)]
)
def test_invalid_transforms(value):
    with pytest.raises(ValueError):
        transform(value)


def test_wrong_lens_model_and_serial_are_rejected(solved):
    rig, data, _, _ = solved
    model = copy.deepcopy(rig["cameras"]["left"]["model"])
    model["intrinsics"]["model"] = "fisheye"
    with pytest.raises(ValueError, match="Convert"):
        camera_model(model)
    samples = copy.deepcopy(data["wrists"]["left"])
    samples[0]["camera_frame"]["serial"] = "wrong-camera"
    with pytest.raises(ValueError, match="binding"):
        solve_wrist(
            samples,
            Board(rig["board"]).object_points,
            camera_model=rig["cameras"]["left"]["model"],
        )


def test_table_requires_normal_and_stationary_board(solved):
    rig = copy.deepcopy(solved[0])
    rig["table"] = dict(enabled=True, thickness_mm=8)
    with pytest.raises(ValueError, match="normal"):
        check_rig(rig)
    pose = np.eye(4)
    other = pose.copy()
    other[0, 3] = 0.03
    with pytest.raises(ValueError, match="moved"):
        table_frame(
            [
                dict(top=pose, left=pose, right=pose),
                dict(top=other, left=other, right=other),
            ],
            dict(thickness_mm=8, board_z_direction="into_table"),
        )


def test_freeze_publish_backup_and_tamper_detection(tmp_path, solved):
    rig, data, _, result = solved
    run, dest = tmp_path / "run", tmp_path / "active.json"
    save_run(run, rig, data, result)
    with pytest.raises(FileExistsError):
        save_run(run, rig, data, result)
    dest.write_text('{"previous":true}')
    publish(run, dest)
    assert json.loads(dest.read_text())["quality_passed"]
    assert len(list(tmp_path.glob("active.json.previous-*.json"))) == 1
    before = dest.read_bytes()
    (run / "observations.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        publish(run, dest)
    assert dest.read_bytes() == before


def test_result_input_binding(tmp_path, solved):
    rig, data, _, result = solved
    altered = dict(data, rig_sha256="modified")
    assert digest(data) != digest(altered)
    with pytest.raises(ValueError, match="match"):
        save_run(tmp_path / "bad", rig, altered, result)
