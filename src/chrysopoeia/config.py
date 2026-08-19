"""Run-config loading (stdlib tomllib) and run-directory bookkeeping."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    path: Path
    raw: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    @property
    def model(self) -> dict[str, Any]:
        return self.section("model")

    @property
    def soak(self) -> dict[str, Any]:
        return self.section("soak")

    @property
    def sft(self) -> dict[str, Any]:
        return self.section("sft")

    @property
    def output(self) -> dict[str, Any]:
        return self.section("output")


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return RunConfig(path=path, raw=raw)


def run_dir(cfg: RunConfig, project_root: Path) -> Path:
    out = cfg.output
    d = project_root / out.get("runs_dir", "runs") / out.get("run_name", "v0")
    d.mkdir(parents=True, exist_ok=True)
    return d
