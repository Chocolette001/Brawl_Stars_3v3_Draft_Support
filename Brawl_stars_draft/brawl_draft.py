#!/usr/bin/env python3
"""
第 6 手推荐。

输入: 我方已锁 2 人 + 敌方已锁 3 人（英文名）。
计算: 先算线性方向 d = K e，再对 d 做 softmax 得到理想成分 x*。
      温度 TEMP 越小越接近纯类型；越大越分散，避免略胜也全押一类。
输出: 在尚未出场、且已填写 (T,A,S,C) 的英雄里，
      按与 x* 的 L1 距离排出最近的 5 个。

克制环: 坦→刺→射→控→坦。投弹手算 S。
已方两人是常数，不改变 x*；只用来排除已出场英雄。
地图/模式偏差：一次逐维缩放 m⊙e 后再算 K，可选预设，默认为不调用。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROSTER_PATH = ROOT / "roster.json"
DIMS = ("T", "A", "S", "C")
TOP_N = 5
# softmax 温度。小 → x* 接近纯类型；大 → 四维摊开。
TEMP = 0.75


@dataclass(frozen=True)
class Vec:
    T: float
    A: float
    S: float
    C: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.T, self.A, self.S, self.C)

    def __add__(self, other: Vec) -> Vec:
        return Vec(self.T + other.T, self.A + other.A, self.S + other.S, self.C + other.C)

    def scale(self, m: Vec) -> Vec:
        """一次偏差：逐维相乘，不改原向量。"""
        return Vec(self.T * m.T, self.A * m.A, self.S * m.S, self.C * m.C)


# K = M - M^T，使 adv(x,y) = x^T K y
# d = K e = (e_A - e_C, e_S - e_T, e_C - e_A, e_T - e_S)
# 模式与掩体合成的一档偏差。1.0 = 不偏。
BIAS_PRESETS: dict[str, tuple[Vec, str]] = {
    "open": (Vec(0.7, 0.8, 1.3, 0.9), "空旷：射手兑现高，坦克被风筝"),
    "walls": (Vec(1.1, 0.8, 1.2, 1.2), "多墙：投弹/控场吃墙，直线切入变差"),
    "grass": (Vec(1.2, 1.3, 0.8, 0.9), "多草：刺客和近战兑现高"),
    "knockout": (Vec(0.7, 1.1, 1.3, 0.9), "击倒/赏金：要杀人，站桩坦较差"),
    "zone": (Vec(1.3, 0.7, 0.9, 1.3), "宝石/热区：占点、控区"),
    "ball": (Vec(1.3, 1.2, 0.8, 0.9), "足球：带球和门前身体"),
    "heist": (Vec(1.3, 0.8, 1.2, 0.8), "保险库：拆门要坦度和远程"),
}

K_LABELS = {
    "T": "需要坦克（对面刺客多于控场）",
    "A": "需要刺客（对面射手多于坦克）",
    "S": "需要射手（对面控场多于刺客）",
    "C": "需要控场（对面坦克多于射手）",
}


def counter_scores(e: Vec) -> Vec:
    """线性克制方向 d = K e，可正可负，不要求和为 1。"""
    T, A, S, C = e.as_tuple()
    return Vec(A - C, S - T, C - A, T - S)


def apply_K(e: Vec, temp: float = TEMP) -> Vec:
    """
    把 K e 变成单纯形上的理想成分。

    先算线性四维，再 softmax：
      x_i = exp(d_i / temp) / sum_j exp(d_j / temp)
    不改 roster。略胜时 x* 不再是 0/1 顶点。
    """
    if temp <= 0:
        raise ValueError("TEMP 必须为正")
    d = counter_scores(e).as_tuple()
    peak = max(d)
    weights = [math.exp((di - peak) / temp) for di in d]
    total = sum(weights)
    return Vec(*(w / total for w in weights))


def team_vec(members: list[Vec]) -> Vec:
    total = Vec(0.0, 0.0, 0.0, 0.0)
    for v in members:
        total = total + v
    return total


def apply_bias(e: Vec, bias: Vec | None) -> Vec:
    """结算一次偏差。bias 为 None 时原样返回。"""
    return e if bias is None else e.scale(bias)


def ideal_vector(
    enemies: list[Vec],
    temp: float = TEMP,
    bias: Vec | None = None,
) -> Vec:
    """第 6 手理想向量 = softmax(K (m⊙e))。"""
    if len(enemies) != 3:
        raise ValueError("需要恰好 3 个敌人向量")
    return apply_K(apply_bias(team_vec(enemies), bias), temp)


def l1(a: Vec, b: Vec) -> float:
    """单纯形上的 L1 距离。越小越接近理想向量。"""
    return sum(abs(x - y) for x, y in zip(a.as_tuple(), b.as_tuple()))


def load_roster(path: Path = ROSTER_PATH) -> dict[str, Vec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    roster: dict[str, Vec] = {}
    for name, stats in raw.items():
        if name.startswith("_") or not isinstance(stats, dict):
            continue
        if any(stats.get(k) is None for k in DIMS):
            continue
        roster[name] = Vec(
            float(stats["T"]),
            float(stats["A"]),
            float(stats["S"]),
            float(stats["C"]),
        )
    return roster


def resolve_name(roster: dict[str, Vec], query: str) -> str:
    """精确匹配（忽略大小写），否则唯一前缀。"""
    q = query.strip()
    if not q:
        raise KeyError("空名字")
    keys = list(roster.keys())
    lower = {n.lower(): n for n in keys}
    if q.lower() in lower:
        return lower[q.lower()]
    prefix = [n for n in keys if n.lower().startswith(q.lower())]
    if len(prefix) == 1:
        return prefix[0]
    if len(prefix) > 1:
        raise KeyError(f"{q!r} 不唯一，可能是: {', '.join(prefix[:8])}")
    raise KeyError(f"{q!r} 不在已填分的英雄里")


def recommend(
    roster: dict[str, Vec],
    allies: list[str],
    enemies: list[str],
    top_n: int = TOP_N,
    bias: Vec | None = None,
) -> tuple[Vec, Vec, list[tuple[str, float, Vec]]]:
    """
    返回 (理想向量 x*, 方向 d=K(m⊙e), 最近的 top_n 个候选)。
    候选排除已出场的 5 人。roster 本身不乘偏差。
    """
    if len(allies) != 2 or len(enemies) != 3:
        raise ValueError("需要 2 个队友名和 3 个敌人名")
    enemy_vecs = [roster[n] for n in enemies]
    e_hat = apply_bias(team_vec(enemy_vecs), bias)
    x_star = apply_K(e_hat)
    d = counter_scores(e_hat)
    taken = set(allies) | set(enemies)
    ranked: list[tuple[str, float, Vec]] = []
    for name, vec in roster.items():
        if name in taken:
            continue
        ranked.append((name, l1(vec, x_star), vec))
    ranked.sort(key=lambda row: (row[1], row[0]))
    return x_star, d, ranked[:top_n]


def fmt_vec(v: Vec) -> str:
    return "  ".join(f"{k}={x:.2f}" for k, x in zip(DIMS, v.as_tuple()))


def read_bias() -> tuple[str, Vec | None]:
    """
    运算开始前问一次。空回车 / 0 / none = 不调用偏差。
    """
    print("偏差预设（只结算一次，作用在敌人合成向量上）:")
    print("  0  none      不调用")
    for i, (key, (vec, note)) in enumerate(BIAS_PRESETS.items(), 1):
        print(f"  {i}  {key:<10} {note}   ({fmt_vec(vec)})")
    keys = list(BIAS_PRESETS.keys())
    while True:
        raw = input("选择偏差 [0]: ").strip().lower()
        if raw in ("", "0", "none", "n", "no"):
            return "none", None
        if raw in BIAS_PRESETS:
            return raw, BIAS_PRESETS[raw][0]
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            key = keys[int(raw) - 1]
            return key, BIAS_PRESETS[key][0]
        print("  输入编号、英文名，或回车跳过")


def read_names(prompt: str, n: int, roster: dict[str, Vec], used: set[str]) -> list[str]:
    while True:
        raw = input(prompt).strip()
        parts = [p for p in raw.replace(",", " ").split() if p]
        if len(parts) != n:
            print(f"  需要恰好 {n} 个英文名，空格分隔")
            continue
        try:
            names = [resolve_name(roster, p) for p in parts]
        except KeyError as exc:
            print(f"  {exc}")
            continue
        if len(set(names)) != n:
            print("  有重复的英雄")
            continue
        if any(n_ in used for n_ in names):
            print("  与已经输入的英雄重复")
            continue
        return names


def main() -> int:
    if not ROSTER_PATH.exists():
        print(f"找不到 {ROSTER_PATH}")
        return 1
    roster = load_roster()
    print(f"已加载 {len(roster)} 名填好向量的英雄（{ROSTER_PATH.name}）")
    if len(roster) < 6:
        print("填分太少，至少需要 5 个已出场 + 若干候选。")
        return 1

    print("名字忽略大小写，可用唯一前缀，例如 Pip = Piper")
    print()
    bias_name, bias = read_bias()
    print()
    allies = read_names("我方两人: ", 2, roster, set())
    enemies = read_names("敌方三人: ", 3, roster, set(allies))

    x_star, d, top = recommend(roster, allies, enemies, bias=bias)
    print()
    print(f"偏差: {bias_name}" + ("" if bias is None else f"   {fmt_vec(bias)}"))
    print(f"我方: {', '.join(allies)}")
    print(f"敌方: {', '.join(enemies)}")
    print(f"理想向量 x*  {fmt_vec(x_star)}")
    print(f"方向   K e   {fmt_vec(d)}")
    peak = max(d.as_tuple())
    reasons = [K_LABELS[k] for k, val in zip(DIMS, d.as_tuple()) if val == peak]
    print("含义: " + "；".join(reasons))
    print()
    if not top:
        print("没有可推荐的未出场英雄（库里填分的人全被选走了）")
        return 0
    print(f"最接近 x* 的 {len(top)} 个:")
    for i, (name, dist, vec) in enumerate(top, 1):
        print(f"  {i}. {name:<16}  L1={dist:.3f}   {fmt_vec(vec)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(130)
