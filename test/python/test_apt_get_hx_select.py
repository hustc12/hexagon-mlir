import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "apt_get_hx_select.py"
SPEC = importlib.util.spec_from_file_location("apt_get_hx_select", MODULE_PATH)
APT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(APT)


def histogram(*peaks):
    result = []
    for cycles, count in peaks:
        result.extend(
            [
                {"cycles": cycles - 10, "count": max(1, count // 5)},
                {"cycles": cycles, "count": count},
                {"cycles": cycles + 10, "count": max(1, count // 4)},
            ]
        )
    return result


def profile(candidate):
    return {
        "schema_version": 1,
        "model": "debug-model",
        "kernel": "encoder",
        "shape": "1x128x768xf16",
        "candidates": [candidate],
    }


def candidate(**updates):
    value = {
        "candidate_id": "tile.0",
        "address_source": "manual",
        "inner_loop": {
            "loop_id": "inner.0",
            "trip_count": 64,
            "iteration_cycle_histogram": histogram((80, 100), (650, 60)),
        },
        "row_bytes": 128,
        "rows": 4,
        "stride": 256,
        "page_split_count": 0,
    }
    value.update(updates)
    return value


def test_selects_paper_like_distance():
    plan = APT.select_plan(profile(candidate()))["plans"][0]
    assert plan["enabled"]
    assert plan["modeled_distance"] == 8
    assert plan["distance"] == 8
    assert plan["injection_site"] == "inner"


def test_unimodal_profile_falls_back():
    value = candidate()
    value["inner_loop"]["iteration_cycle_histogram"] = histogram((100, 50))
    plan = APT.select_plan(profile(value))["plans"][0]
    assert not plan["enabled"]
    assert plan["reason"] == "no_separable_latency_peaks"


def test_short_inner_loop_selects_profiled_outer_site():
    value = candidate()
    value["inner_loop"]["trip_count"] = 1
    value["outer_loop"] = {
        "loop_id": "outer.0",
        "trip_count": 32,
        "iteration_cycle_histogram": histogram((200, 80), (1000, 40)),
    }
    plan = APT.select_plan(profile(value))["plans"][0]
    assert plan["enabled"]
    assert plan["injection_site"] == "outer"
    assert plan["distance"] == 4


def test_capacity_projects_distance_down():
    plan = APT.select_plan(
        profile(candidate(residency_budget_bytes=2048))
    )["plans"][0]
    assert plan["enabled"]
    assert plan["distance"] == 4
    assert plan["projected_live_bytes"] == 2048


def test_shape_mismatch_disables_entire_plan():
    plan = APT.select_plan(profile(candidate()), expected_shape="different")
    assert plan["status"] == "no_prefetch"
    assert plan["reason"] == "shape_mismatch"
    assert plan["plans"] == []
