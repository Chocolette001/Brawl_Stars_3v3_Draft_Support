#!/usr/bin/env python3
"""
初始化 roster.json 的英文名，并循环询问尚未填写的 (T, A, S, C)。

输入为 100 倍整数/小数，程序除以 100。
例: 5 10 80 5  ->  T=0.05 A=0.10 S=0.80 C=0.05
已有四个数字的条目不会被覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROSTER_PATH = ROOT / "roster.json"

# 英文名，与游戏内常用写法对齐。新英雄可直接加在这张表末尾。
BRAWLERS = [
    "8-Bit",
    "Alli",
    "Amber",
    "Angelo",
    "Ash",
    "Barley",
    "Bea",
    "Belle",
    "Berry",
    "Bibi",
    "Bo",
    "Bolt",
    "Bonnie",
    "Brock",
    "Bull",
    "Buster",
    "Buzz",
    "Byron",
    "Carl",
    "Charlie",
    "Chester",
    "Chuck",
    "Clancy",
    "Colette",
    "Colt",
    "Cordelius",
    "Cosmo",
    "Crow",
    "Damian",
    "Darryl",
    "Doug",
    "Draco",
    "Dynamike",
    "Edgar",
    "El Primo",
    "Emz",
    "Eve",
    "Fang",
    "Finx",
    "Frank",
    "Gale",
    "Gene",
    "Gigi",
    "Glowy",
    "Gray",
    "Griff",
    "Grom",
    "Gus",
    "Hank",
    "Jacky",
    "Jae-yong",
    "Janet",
    "Jessie",
    "Juju",
    "Kaze",
    "Kenji",
    "Kit",
    "Larry & Lawrie",
    "Leon",
    "Lily",
    "Lola",
    "Lou",
    "Lumi",
    "Maisie",
    "Mandy",
    "Max",
    "Meeple",
    "Meg",
    "Melodie",
    "Mico",
    "Mina",
    "Moe",
    "Mortis",
    "Mr. P",
    "Najia",
    "Nani",
    "Nita",
    "Nori",
    "Ollie",
    "Otis",
    "Pam",
    "Pearl",
    "Penny",
    "Pierce",
    "Piper",
    "Poco",
    "R-T",
    "Rico",
    "Rosa",
    "Ruffs",
    "Sam",
    "Sandy",
    "Shade",
    "Shelly",
    "Sirius",
    "Spike",
    "Sprout",
    "Squeak",
    "Starr Nova",
    "Stu",
    "Surge",
    "Tara",
    "Tick",
    "Trunk",
    "Vince",
    "Wendy",
    "Willow",
    "Ziggy",
]

EMPTY = {"T": None, "A": None, "S": None, "C": None}
KEYS = ("T", "A", "S", "C")


def is_filled(stats: object) -> bool:
    if not isinstance(stats, dict):
        return False
    return all(stats.get(k) is not None for k in KEYS)


def save(roster: dict) -> None:
    ROSTER_PATH.write_text(
        json.dumps(roster, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_line(line: str) -> dict[str, float] | None:
    """
    一行四个数，顺序 T A S C。
    输入的是 100 倍（95 表示 0.95），这里除以 100。
    """
    parts = line.replace(",", " ").split()
    if len(parts) != 4:
        return None
    try:
        raw = [float(p) for p in parts]
    except ValueError:
        return None
    return {k: v / 100.0 for k, v in zip(KEYS, raw)}


def ensure_names(existing: dict) -> dict:
    """先保证所有英文名都在文件里，已填的数字不覆盖。"""
    roster = {}
    for name in BRAWLERS:
        roster[name] = existing[name] if is_filled(existing.get(name)) else dict(EMPTY)
    return roster


def pending_names(roster: dict) -> list[str]:
    return [name for name in BRAWLERS if not is_filled(roster.get(name))]


def main() -> None:
    existing = {}
    if ROSTER_PATH.exists():
        existing = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))

    roster = ensure_names(existing)
    save(roster)

    todo = pending_names(roster)
    print(f"{ROSTER_PATH.name}: {len(roster)} names, {len(todo)} unfilled")
    print("each line: T A S C   (100x, e.g. 5 10 80 5  ->  0.05 0.10 0.80 0.05)")
    print("empty / s = skip this brawler, q = quit")
    print()

    for i, name in enumerate(todo, 1):
        while True:
            line = input(f"[{i}/{len(todo)}] {name}  T A S C > ").strip()
            if line.lower() in ("", "s", "skip"):
                print(f"  skipped {name}")
                break
            if line.lower() in ("q", "quit"):
                save(roster)
                print("saved, bye")
                return
            vec = parse_line(line)
            if vec is None:
                print("  need four numbers, example: 0 90 10 0")
                continue
            total = sum(vec.values())
            if abs(total - 1.0) > 0.02:
                print(f"  warning: components sum to {total:.2f} (expected ~1 after /100)")
            roster[name] = vec
            save(roster)
            print(
                f"  saved {name} = "
                + " ".join(f"{k}={vec[k]:.2f}" for k in KEYS)
            )
            break

    left = pending_names(roster)
    print(f"done. filled {len(roster) - len(left)}, still empty {len(left)}")


if __name__ == "__main__":
    main()

