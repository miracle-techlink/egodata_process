import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from src.pose6d import extract_pose6d as ep


class StubDepthModel:
    def __init__(self, source, **config):
        self.source = source
        self.config = config
        self.load_state_dict = Mock()
        self.cuda = Mock(return_value=self)
        self.eval = Mock(return_value=self)


@pytest.fixture(autouse=True)
def stub_runtime(monkeypatch):
    saved_modules = {
        name: module for name, module in sys.modules.items()
        if name.startswith(("depth_anything_v2", "_egodata_depth_anything_v2_"))
    }
    monkeypatch.setattr(sys, "path", list(sys.path))
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    torch.load = Mock(return_value={"stub_weights": 1})
    torch.StubDepthModel = StubDepthModel
    monkeypatch.setitem(sys.modules, "torch", torch)
    yield torch
    for name in list(sys.modules):
        if name.startswith(("depth_anything_v2", "_egodata_depth_anything_v2_")):
            del sys.modules[name]
    sys.modules.update(saved_modules)


def write_stub_repo(repo):
    for metric in (False, True):
        source = "metric" if metric else "relative"
        model_root = repo / "metric_depth" if metric else repo
        package = model_root / "depth_anything_v2"
        package.mkdir(parents=True)
        (package / "backend.py").write_text(
            f"from torch import StubDepthModel\nSOURCE = {source!r}\n"
        )
        parameter = ", max_depth" if metric else ""
        argument = ", max_depth=max_depth" if metric else ""
        (package / "dpt.py").write_text(
            "from .backend import StubDepthModel, SOURCE\n"
            "class DepthAnythingV2(StubDepthModel):\n"
            f"    def __init__(self, encoder, features, out_channels{parameter}):\n"
            "        super().__init__(source=SOURCE, encoder=encoder, "
            f"features=features, out_channels=out_channels{argument})\n"
        )
    return repo


@pytest.fixture
def depth_repo(tmp_path):
    return write_stub_repo(tmp_path / "depth-anything")


@pytest.mark.parametrize("metric", [False, True])
@pytest.mark.parametrize("encoder", ["vits", "vitb", "vitl"])
def test_constructor_selects_requested_model(depth_repo, stub_runtime, metric, encoder):
    provider = ep.DepthAnythingProvider(
        str(depth_repo), "weights.pth", encoder=encoder,
        input_size=256, metric=metric, max_depth=80.0,
    )

    assert provider.model.source == ("metric" if metric else "relative")
    expected = dict(ep._DA2_CONFIGS[encoder])
    if metric:
        expected["max_depth"] = 80.0
    assert provider.model.config == expected
    assert provider.input_size == 256
    stub_runtime.load.assert_called_once_with("weights.pth", map_location="cpu")
    provider.model.load_state_dict.assert_called_once_with({"stub_weights": 1})
    provider.model.cuda.assert_called_once_with()
    provider.model.eval.assert_called_once_with()


@pytest.mark.parametrize("first_metric", [False, True])
def test_metric_and_relative_coexist_in_either_order(depth_repo, first_metric):
    original_path = list(sys.path)
    modes = [first_metric, not first_metric, first_metric]
    providers = [
        ep.DepthAnythingProvider(str(depth_repo), "weights.pth", metric=metric)
        for metric in modes
    ]

    for provider, metric in zip(providers, modes):
        assert provider.model.source == ("metric" if metric else "relative")
        assert ("max_depth" in provider.model.config) is metric
        if metric:
            assert provider.model.config["max_depth"] == 20.0
    assert type(providers[0].model) is not type(providers[1].model)
    assert type(providers[0].model) is type(providers[2].model)
    assert sys.path == original_path


@pytest.mark.parametrize("metric", [False, True])
def test_ignores_cached_same_name_package(depth_repo, monkeypatch, metric):
    cached_root = depth_repo if metric else depth_repo / "metric_depth"
    monkeypatch.syspath_prepend(str(cached_root))
    cached_module = importlib.import_module("depth_anything_v2.dpt")

    provider = ep.DepthAnythingProvider(str(depth_repo), "weights.pth", metric=metric)

    assert provider.model.source == ("metric" if metric else "relative")
    assert sys.modules["depth_anything_v2.dpt"] is cached_module
    assert type(provider.model) is not cached_module.DepthAnythingV2


@pytest.mark.parametrize("metric", [False, True])
def test_separate_repositories_do_not_share_cached_models(depth_repo, tmp_path, metric):
    other_repo = write_stub_repo(tmp_path / "other-depth-anything")
    first = ep.DepthAnythingProvider(str(depth_repo), "weights.pth", metric=metric)
    second = ep.DepthAnythingProvider(str(other_repo), "weights.pth", metric=metric)

    assert first.model.source == second.model.source == ("metric" if metric else "relative")
    assert type(first.model) is not type(second.model)


@pytest.mark.parametrize("failure", ["missing_model", "broken_import"])
def test_invalid_metric_repo_never_falls_back(depth_repo, monkeypatch, stub_runtime, failure):
    monkeypatch.syspath_prepend(str(depth_repo))
    cached_module = importlib.import_module("depth_anything_v2.dpt")
    metric_model = depth_repo / "metric_depth" / "depth_anything_v2" / "dpt.py"
    if failure == "missing_model":
        metric_model.unlink()
        error, message = RuntimeError, "metric_depth.*dpt.py"
    else:
        metric_model.write_text("raise ImportError('missing metric dependency')\n")
        error, message = ImportError, "missing metric dependency"

    with pytest.raises(error, match=message):
        ep.DepthAnythingProvider(str(depth_repo), "weights.pth", metric=True)

    stub_runtime.load.assert_not_called()
    assert sys.modules["depth_anything_v2.dpt"] is cached_module


@pytest.mark.parametrize("strict", [False, True])
def test_invalid_metric_repo_preserves_strict_behavior(depth_repo, strict):
    (depth_repo / "metric_depth" / "depth_anything_v2" / "dpt.py").unlink()
    args = SimpleNamespace(
        depth_source="depth_anything", depth_anything_repo=str(depth_repo),
        depth_anything_ckpt="weights.pth", depth_anything_encoder="vitl",
        depth_anything_input_size=518, depth_anything_metric=True,
        depth_anything_max_depth=20.0, strict=strict,
    )
    warnings, flags = [], []

    if strict:
        with pytest.raises(RuntimeError, match="metric_depth.*dpt.py"):
            ep.build_depth_provider(args, warnings, flags)
        assert warnings == flags == []
    else:
        provider = ep.build_depth_provider(args, warnings, flags)
        assert isinstance(provider, ep.NullDepthProvider)
        assert flags == ["depth_anything_unavailable"]
        assert len(warnings) == 1
        assert "depth_anything_unavailable" in warnings[0]
        assert "metric_depth" in warnings[0]
