"""同位素名义质量峰分布（isotopic envelope）的精确有理数计算。

给定化学式中每种元素的原子数（由 :mod:`app.parser` 解析得到）与每种元素的
若干同位素（正整数名义质量数 + 非负丰度），把该元素的每个原子视为一次
**独立抽样**：

* 枚举 ``n`` 个原子在 ``k`` 种同位素上的全部占有数组合
  ``(c_1, …, c_k)``（满足 ``Σc_i = n``）；
* 每个组合的概率取多项式分布
  ``n!/(c_1!…c_k!) · Π p_i^c_i``，名义质量取 ``Σ c_i·m_i``；
* **同质量的不同组合概率相加（合并）**；
* 多种元素之间做质量分布的卷积（独立事件相乘、同质量相加），得到整个
  化学式的全部名义质量峰。

丰度允许为 0：占有数组合一律完整枚举，**不在合并前以“低/零概率”为由
截断**——提前剪枝会漏掉与其它组合同质量相加的项，使合并结果错误。在
原子数 ≤ 30、每组同位素 ≤ 3 的限制下，单元素组合数至多
``C(32,2) = 496``，全枚举成本可忽略。

全部运算使用 :class:`fractions.Fraction`（不经过 float），概率以最简
分数字符串（``"p/q"``、``"1"``、``"0"``）输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import Iterator


@dataclass(frozen=True)
class Isotope:
    """一种同位素：名义质量数（正整数）与非负丰度（精确有理数）。"""

    mass_number: int
    abundance: Fraction


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """枚举把 ``total`` 个相同原子分到 ``parts`` 种同位素上的全部占有数。

    生成满足 ``c_1 + … + c_parts = total`` 的全部非负整数元组
    （弱组合，共 ``C(total + parts - 1, parts - 1)`` 个）。
    """
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def element_envelope(
    atom_count: int, isotopes: list[Isotope]
) -> dict[int, Fraction]:
    """单元素 ``atom_count`` 个独立原子的名义质量分布（同质量已合并）。"""
    masses = [iso.mass_number for iso in isotopes]
    abundances = [iso.abundance for iso in isotopes]
    peaks: dict[int, Fraction] = {}
    for counts in _compositions(atom_count, len(isotopes)):
        mass = 0
        ways = 1
        remaining = atom_count
        probability = Fraction(1)
        for c, m, p in zip(counts, masses, abundances):
            mass += c * m
            ways *= comb(remaining, c)  # 多项式系数 = Π C(剩余, c_i)
            remaining -= c
            probability *= p**c
        # 同质量组合合并：概率精确相加（零概率组合也保留，不截断）
        peaks[mass] = peaks.get(mass, Fraction(0)) + ways * probability
    return peaks


def compute_envelope(
    composition: dict[str, int], isotope_groups: dict[str, list[Isotope]]
) -> list[tuple[int, Fraction]]:
    """计算整个化学式的名义质量峰，返回 ``[(质量, 概率), …]``（质量升序）。

    ``composition`` 为元素 -> 原子数；``isotope_groups`` 必须恰好覆盖其中
    每种元素。不同元素的抽样相互独立，故逐元素做分布卷积。
    """
    envelope: dict[int, Fraction] = {0: Fraction(1)}
    for element in sorted(composition):
        block = element_envelope(composition[element], isotope_groups[element])
        merged: dict[int, Fraction] = {}
        for mass_left, prob_left in envelope.items():
            for mass_right, prob_right in block.items():
                key = mass_left + mass_right
                merged[key] = merged.get(key, Fraction(0)) + prob_left * prob_right
        envelope = merged
    return sorted(envelope.items())


def most_intense_peak(
    peaks: list[tuple[int, Fraction]],
) -> tuple[int, Fraction]:
    """返回概率最高的峰；概率并列时取质量较小者（峰已按质量升序）。"""
    best_mass: int | None = None
    best_probability: Fraction | None = None
    for mass, probability in peaks:
        if best_probability is None or probability > best_probability:
            best_mass, best_probability = mass, probability
    assert best_mass is not None and best_probability is not None
    return best_mass, best_probability
