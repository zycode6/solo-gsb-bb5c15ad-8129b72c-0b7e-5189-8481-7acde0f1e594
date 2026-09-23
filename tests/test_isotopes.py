"""POST /api/v1/isotopes/envelope 的 HTTP API 与核心算法测试。

覆盖：单/多元素多项式分布、同质量组合合并（多同位素可出现相同质量和）、
零丰度峰保留（合并前不截断）、并列最高峰取质量较小者、最简分数字符串、
括号化学式计数、离子电荷不影响质量分布、全部峰概率之和为 1、规模上限
（3 元素 / 30 原子 / 每组 3 同位素）；以及 400 化学式解析失败与 422
INVALID_ISOTOPES（电子物种、元素/原子数超限、缺失/多余/未知元素、
质量数非正或重复、丰度非法或和不为 1、JSON 数字、空组等）。
"""

from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.isotopes import (
    Isotope,
    compute_envelope,
    element_envelope,
    most_intense_peak,
)
from app.main import app

client = TestClient(app)

PATH = "/api/v1/isotopes/envelope"


def iso(mass_number, abundance):
    return {"mass_number": mass_number, "abundance": abundance}


def body(formula, groups):
    return {"formula": formula, "isotopes": groups}


def group(*pairs):
    return [iso(mass, ab) for mass, ab in pairs]


# ------------------------------------------------------------ 核心算法单元测试
def test_element_envelope_single_isotope_is_certain():
    peaks = element_envelope(3, [Isotope(12, Fraction(1))])
    assert peaks == {36: Fraction(1)}


def test_element_envelope_binomial_coefficients():
    # Cl2: 35 @ 3/4、37 @ 1/4 -> 70:9/16、72:6/16、74:1/16
    peaks = element_envelope(
        2, [Isotope(35, Fraction(3, 4)), Isotope(37, Fraction(1, 4))]
    )
    assert peaks == {
        70: Fraction(9, 16),
        72: Fraction(6, 16),
        74: Fraction(1, 16),
    }


def test_element_envelope_trinomial_merges_same_mass_combinations():
    # 质量数 1 @ 1/3、2 @ 1/3、3 @ 1/3，n=2：
    # 质量 3 可由 (1,2)/(2,1) 两个组合给出，合并后为 4/9 而非 2/9
    peaks = element_envelope(
        2,
        [Isotope(1, Fraction(1, 3)), Isotope(2, Fraction(1, 3)),
         Isotope(3, Fraction(1, 3))],
    )
    assert peaks == {
        2: Fraction(1, 9),
        3: Fraction(2, 9),
        4: Fraction(3, 9),
        5: Fraction(2, 9),
        6: Fraction(1, 9),
    }
    assert sum(peaks.values()) == 1


def test_zero_abundance_combinations_are_not_dropped_before_merge():
    # 同位素 1 丰度 1、2 丰度 0：质量 3、4 的组合概率为 0 仍必须枚举返回
    peaks = element_envelope(2, [Isotope(1, Fraction(1)), Isotope(2, Fraction(0))])
    assert peaks == {2: Fraction(1), 3: Fraction(0), 4: Fraction(0)}


def test_zero_abundance_still_contributes_to_shared_mass_merge():
    # 质量 1/2 丰度 1/0 与质量 2/3 丰度 0/1：质量 3 在组内有
    # (1,2) 零概率 + (3,0) 零概率 -> 0；质量 4 由 (2,2) 与 (1,3)/(3,1)
    # 汇聚，验证合并发生在全枚举之后
    peaks = element_envelope(
        2,
        [Isotope(1, Fraction(1, 2)), Isotope(2, Fraction(0)),
         Isotope(3, Fraction(1, 2))],
    )
    assert peaks == {2: Fraction(1, 4), 3: Fraction(0), 4: Fraction(1, 2),
                    5: Fraction(0), 6: Fraction(1, 4)}


def test_compute_envelope_convolves_independent_elements():
    # H2O：H(1/2 各半) × O(16/18 各半)
    peaks = compute_envelope(
        {"H": 2, "O": 1},
        {
            "H": [Isotope(1, Fraction(1, 2)), Isotope(2, Fraction(1, 2))],
            "O": [Isotope(16, Fraction(1, 2)), Isotope(18, Fraction(1, 2))],
        },
    )
    expected = [(18, Fraction(1, 8)), (19, Fraction(1, 4)), (20, Fraction(1, 4)),
                (21, Fraction(1, 4)), (22, Fraction(1, 8))]
    assert peaks == expected
    assert [m for m, _ in peaks] == sorted(m for m, _ in peaks)


def test_most_intense_peak_tie_prefers_smaller_mass():
    peaks = [(10, Fraction(1, 4)), (11, Fraction(1, 2)), (12, Fraction(1, 4))]
    assert most_intense_peak(peaks) == (11, Fraction(1, 2))
    tied = [(1, Fraction(1, 2)), (2, Fraction(1, 2))]
    assert most_intense_peak(tied) == (1, Fraction(1, 2))


# ------------------------------------------------------------ HTTP 成功路径
def test_simple_diatomic_cl2():
    resp = client.post(
        PATH,
        json=body("Cl2", {"Cl": group((35, "0.75"), (37, "0.25"))}),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["input"] == "Cl2"
    assert data["normalized"] == "Cl2"
    assert data["total_atoms"] == 2
    assert data["peaks"] == [
        {"mass": 70, "probability": "9/16"},
        {"mass": 72, "probability": "3/8"},
        {"mass": 74, "probability": "1/16"},
    ]
    assert data["most_intense_peak"] == {"mass": 70, "probability": "9/16"}


def test_water_full_envelope_fractions():
    resp = client.post(
        PATH,
        json=body(
            "H2O",
            {
                "H": group((1, "0.5"), (2, "0.5")),
                "O": group((16, "0.5"), (18, "0.5")),
            },
        ),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert [p["mass"] for p in data["peaks"]] == [18, 19, 20, 21, 22]
    assert [p["probability"] for p in data["peaks"]] == [
        "1/8", "1/4", "1/4", "1/4", "1/8"
    ]
    # 质量 19/20/21 并列 1/4，取质量最小者
    assert data["most_intense_peak"] == {"mass": 19, "probability": "1/4"}


def test_probabilities_rendered_as_reduced_fraction_strings():
    resp = client.post(
        PATH, json=body("H4", {"H": group((1, "0.5"), (2, "0.5"))})
    )
    data = resp.json()
    probs = {p["mass"]: p["probability"] for p in data["peaks"]}
    assert probs == {4: "1/16", 5: "1/4", 6: "3/8", 7: "1/4", 8: "1/16"}
    # 6/16 必须约分为 3/8
    assert probs[6] == "3/8"


def test_certain_isotope_returns_one_peak_with_probability_one():
    resp = client.post(
        PATH,
        json=body("CO2", {
            "C": group((12, "1")),
            "O": group((16, "1")),
        }),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["peaks"] == [{"mass": 44, "probability": "1"}]
    assert data["most_intense_peak"] == {"mass": 44, "probability": "1"}


def test_zero_abundance_peaks_are_retained():
    resp = client.post(
        PATH, json=body("H2", {"H": group((1, "1"), (2, "0"))})
    )
    data = resp.json()
    assert data["peaks"] == [
        {"mass": 2, "probability": "1"},
        {"mass": 3, "probability": "0"},
        {"mass": 4, "probability": "0"},
    ]
    assert data["most_intense_peak"] == {"mass": 2, "probability": "1"}


def test_brackets_are_counted_by_existing_parser():
    # Ca(OH)2：Ca 1、O 2、H 2，全部单一同位素 -> 40 + 2*16 + 2*1 = 74
    resp = client.post(
        PATH,
        json=body("Ca(OH)2", {
            "Ca": group((40, "1")),
            "O": group((16, "1")),
            "H": group((1, "1")),
        }),
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["normalized"] == "Ca(OH)2"
    assert data["total_atoms"] == 5
    assert data["peaks"] == [{"mass": 74, "probability": "1"}]


def test_charge_does_not_affect_envelope():
    neutral = client.post(PATH, json=body("SO4", {
        "S": group((32, "1")), "O": group((16, "1"))})).json()
    ion = client.post(PATH, json=body("SO4^2-", {
        "S": group((32, "1")), "O": group((16, "1"))})).json()
    assert ion["normalized"] == "SO4^2-"
    assert ion["peaks"] == neutral["peaks"] == [
        {"mass": 96, "probability": "1"}]


def test_decimal_abundances_exact_sum_one_accepted():
    # 0.9893 + 0.0107 = 1.0000（十进制精确），2 个 C 给出 13C2 峰
    resp = client.post(
        PATH,
        json=body("C2", {"C": group((12, "0.9893"), (13, "0.0107"))}),
    )
    assert resp.status_code == 200, resp.text
    probs = {p["mass"]: p["probability"] for p in resp.json()["peaks"]}
    assert Fraction(probs[24]) == Fraction("0.9893") ** 2
    assert Fraction(probs[25]) == 2 * Fraction("0.9893") * Fraction("0.0107")
    assert Fraction(probs[26]) == Fraction("0.0107") ** 2


def test_three_isotopes_three_elements_at_limits():
    # 3 种元素、每组 3 同位素的接口上限（HOC 三种元素各 1 个原子）
    resp = client.post(
        PATH,
        json=body("HOC", {
            "H": group((1, "0.9"), (2, "0.05"), (3, "0.05")),
            "O": group((16, "0.9"), (17, "0.05"), (18, "0.05")),
            "C": group((12, "0.9"), (13, "0.05"), (14, "0.05")),
        }),
    )
    assert resp.status_code == 200, resp.text
    peaks = resp.json()["peaks"]
    assert sum(Fraction(p["probability"]) for p in peaks) == 1
    # 每种元素各 1 个原子、质量数间距均为 1：三元素质量和大量同质量碰撞，
    # 27 个组合合并为质量 29～35 共 7 个峰（系数 1,3,6,7,6,3,1）
    assert [p["mass"] for p in peaks] == [29, 30, 31, 32, 33, 34, 35]
    assert [Fraction(p["probability"]) for p in peaks] == [
        Fraction(729, 1000), Fraction(243, 2000), Fraction(513, 4000),
        Fraction(109, 8000), Fraction(57, 8000), Fraction(3, 8000),
        Fraction(1, 8000),
    ]


def test_total_atoms_limit_30_performance_and_completeness():
    # C30 + 两种同位素：31 个峰，概率之和精确为 1
    resp = client.post(
        PATH,
        json=body("C30", {"C": group((12, "0.989"), (13, "0.011"))}),
    )
    assert resp.status_code == 200, resp.text
    peaks = resp.json()["peaks"]
    assert len(peaks) == 31
    assert [p["mass"] for p in peaks] == list(range(360, 391))
    assert sum(Fraction(p["probability"]) for p in peaks) == 1
    # 最重峰（全部 13C）概率 = 0.011^30，是最简分数
    assert Fraction(peaks[-1]["probability"]) == Fraction("0.011") ** 30


def test_three_isotopes_thirty_atoms_enumeration_budget():
    # 最坏枚举规模：30 原子 × 3 同位素 = C(32,2)=496 个占有数组合
    resp = client.post(
        PATH,
        json=body("C30", {"C": group(
            (12, "0.989"), (13, "0.0109"), (14, "0.0001"))}),
    )
    assert resp.status_code == 200, resp.text
    peaks = resp.json()["peaks"]
    assert sum(Fraction(p["probability"]) for p in peaks) == 1


def test_response_shape():
    resp = client.post(PATH, json=body("H", {"H": group((1, "1"))}))
    data = resp.json()
    assert set(data) == {
        "input", "normalized", "total_atoms", "peaks", "most_intense_peak"
    }
    assert set(data["peaks"][0]) == {"mass", "probability"}
    assert set(data["most_intense_peak"]) == {"mass", "probability"}


def test_stateless_same_request_same_result():
    payload = body("Cl2", {"Cl": group((35, "0.75"), (37, "0.25"))})
    first = client.post(PATH, json=payload)
    second = client.post(PATH, json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()


def test_scientific_notation_abundances_accepted():
    resp = client.post(
        PATH,
        json=body("C", {"C": group((12, "9.893e-1"), (13, "1.07e-2"))}),
    )
    assert resp.status_code == 200, resp.text
    peaks = resp.json()["peaks"]
    assert sum(Fraction(p["probability"]) for p in peaks) == 1


# ------------------------------------------------------------ 400 化学式解析失败
@pytest.mark.parametrize("formula", ["H2(", "Xy", ""] )
def test_unparseable_formula_returns_400(formula):
    resp = client.post(
        PATH, json=body(formula, {"H": group((1, "1"))})
    )
    assert resp.status_code == (400 if formula else 422)
    if formula:
        assert resp.json()["error"]["code"] == "FORMULA_PARSE_ERROR"
        assert resp.json()["error"]["details"]["field"] == "formula"


# ------------------------------------------------------------ 422 INVALID_ISOTOPES
def test_electron_species_rejected():
    for text in ("e-", "e+", "e"):
        resp = client.post(PATH, json=body(text, {}))
        assert resp.status_code == 422, text
        err = resp.json()["error"]
        assert err["code"] == "INVALID_ISOTOPES"
        assert err["details"]["electron_species"].startswith("e")


def test_more_than_three_elements_rejected():
    resp = client.post(
        PATH,
        json=body("CHClBr", {
            "C": group((12, "1")), "H": group((1, "1")),
            "Cl": group((35, "1")), "Br": group((79, "1")),
        }),
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["too_many_elements"] == ["Br", "C", "Cl", "H"]


def test_more_than_thirty_total_atoms_rejected():
    resp = client.post(
        PATH, json=body("C31", {"C": group((12, "1"))})
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["details"]["too_many_atoms"] == 31


def test_missing_element_group_rejected():
    resp = client.post(
        PATH, json=body("H2O", {"H": group((1, "1"))})
    )
    assert resp.status_code == 422
    err = resp.json()["error"]
    assert err["code"] == "INVALID_ISOTOPES"
    assert err["details"]["missing_elements"] == ["O"]


def test_extra_element_not_in_formula_rejected():
    resp = client.post(
        PATH, json=body("H2", {
            "H": group((1, "1")), "O": group((16, "1"))})
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["extra_elements"] == ["O"]
    assert "unknown_elements" not in details


def test_unknown_element_symbol_rejected():
    resp = client.post(
        PATH, json=body("H2", {
            "H": group((1, "1")), "X": group((1, "1"))})
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["unknown_elements"] == ["X"]


def test_more_than_three_isotopes_in_group_rejected():
    resp = client.post(
        PATH,
        json=body("H", {"H": [
            iso(1, "0.25"), iso(2, "0.25"), iso(3, "0.25"), iso(4, "0.25")]}),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_ISOTOPES"
    assert resp.json()["error"]["details"]["too_many_isotopes"] == ["H"]


def test_duplicate_mass_numbers_rejected():
    resp = client.post(
        PATH, json=body("H", {"H": group((1, "0.5"), (1, "0.5"))})
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["duplicate_mass_numbers"] == [
        {"element": "H", "mass_number": 1}]


@pytest.mark.parametrize("bad_mass", [0, -12])
def test_non_positive_mass_number_rejected(bad_mass):
    resp = client.post(
        PATH, json=body("H", {"H": group((bad_mass, "1"))})
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["non_positive_mass_numbers"] == [
        {"element": "H", "mass_number": bad_mass}]


@pytest.mark.parametrize("bad", ["-1", "-0.0", "NaN", "Infinity", "abc"])
def test_invalid_abundance_rejected(bad):
    resp = client.post(
        PATH, json=body("H", {"H": group((1, bad))})
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["invalid_abundances"] == [
        {"element": "H", "mass_number": 1, "abundance": bad}]


@pytest.mark.parametrize("abundances,expected_sum", [
    ((("1", "0.1"), ("2", "0.2")), "0.3"),        # 0.1 + 0.2 = 0.3 ≠ 1（十进制精确判定）
    ((("1", "0.9999"), ("2", "0.00009")), "0.99999"),  # 差 0.00001
    ((("1", "0.5"), ("2", "0.25")), "0.75"),
])
def test_abundance_sum_not_exactly_one_rejected(abundances, expected_sum):
    resp = client.post(
        PATH,
        json=body("H", {"H": [iso(m, a) for m, a in abundances]}),
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["abundance_sum_not_one"] == [
        {"element": "H", "sum": expected_sum}]


def test_multiple_problems_aggregated():
    resp = client.post(
        PATH,
        json=body("H2O", {
            "H": [iso(1, "0.5"), iso(1, "0.5")],   # 质量数重复
            "X": [iso(1, "1")],                     # 未知元素
            # 缺少 O
        }),
    )
    assert resp.status_code == 422
    details = resp.json()["error"]["details"]
    assert details["duplicate_mass_numbers"] == [
        {"element": "H", "mass_number": 1}]
    assert details["unknown_elements"] == ["X"]
    assert details["missing_elements"] == ["O"]


# ------------------------------------------------------------ 422 INVALID_REQUEST（结构）
def test_json_number_abundance_is_structural_error():
    resp = client.post(PATH, json={
        "formula": "H",
        "isotopes": {"H": [{"mass_number": 1, "abundance": 1.0}]}})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_float_mass_number_is_structural_error():
    resp = client.post(PATH, json={
        "formula": "H",
        "isotopes": {"H": [iso(1.5, "1")]}})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_isotope_group_is_structural_error():
    resp = client.post(PATH, json=body("H", {"H": []}))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_missing_formula_field_is_structural_error():
    resp = client.post(PATH, json={"isotopes": {}})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_blank_formula_is_structural_error():
    resp = client.post(PATH, json=body("   ", {}))
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_isotopes_not_an_object_is_structural_error():
    resp = client.post(PATH, json={"formula": "H", "isotopes": []})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
