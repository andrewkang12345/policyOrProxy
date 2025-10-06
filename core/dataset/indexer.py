"""Episode index helpers for the generated datasets."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class EpisodeRecord:
    path: str
    length: int
    split: str
    policy_id: Optional[str] = None
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodeIndexer:
    root: Path
    entries: List[EpisodeRecord] = field(default_factory=list)

    def add(self, record: EpisodeRecord) -> None:
        self.entries.append(record)

    def add_episode(self, split: str, path: Path, length: int, policy_id: Optional[str] = None, meta: Optional[Dict[str, float]] = None) -> None:
        rel = path.relative_to(self.root)
        record = EpisodeRecord(path=str(rel), length=length, split=split, policy_id=policy_id, meta=meta or {})
        self.add(record)

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.root / "index.json"
        payload = [record.__dict__ for record in self.entries]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    @classmethod
    def load(cls, root: Path, path: Optional[Path] = None) -> "EpisodeIndexer":
        target = path or root / "index.json"
        with target.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        entries = [EpisodeRecord(**item) for item in payload]
        return cls(root=root, entries=entries)

    def iter_split(self, split: str) -> Iterator[EpisodeRecord]:
        for entry in self.entries:
            if entry.split == split:
                yield entry

    def splits(self) -> List[str]:
        return sorted({entry.split for entry in self.entries})

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.entries:
            counts.setdefault(entry.split, 0)
            counts[entry.split] += entry.length
        return counts
