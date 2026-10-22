from pathlib import Path

from src.pose6d.run_human_rgbd import build_parser


def test_published_baseline_configuration():
    root = Path(__file__).resolve().parents[1]
    tokens = (root / 'configs/human_rgbd_baseline.args').read_text().splitlines()
    args = build_parser().parse_args(tokens + [
        '--foundationpose-repo', '/example/FoundationPose',
        '--scene-dir', '/example/scene', '--mesh-file', '/example/mesh.obj',
        '--out-dir', '/example/new_run', '--human-evidence', '/example/source.json',
        '--fps-source', '/example/official_timing.json',
    ])
    assert args.source_fps == 30
    assert args.depth_scale == 0.001
    assert args.est_refine_iter == 5
    assert args.track_refine_iter == 1
    assert args.hand_model_complexity == 0
    assert args.torch_threads == args.opencv_threads == 1
    assert args.prefetch_input and args.pace_replay
    assert args.limit_frames is None
    assert not args.input_mirrored
