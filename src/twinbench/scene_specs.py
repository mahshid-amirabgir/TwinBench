"""Scene and workspace specifications."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import copy
import yaml
import numpy as np

from twinbench.ids import canon_robot_id, canon_scene_id


def _load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None or not isinstance(obj, dict):
        raise ValueError(f"Bad YAML mapping: {path}")
    return obj


def find_repo_root(start: str | Path | None = None) -> Path:
    p = Path(start).resolve() if start else Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "robot_specs").exists() and (parent / "scenes").exists():
            return parent
    raise FileNotFoundError("Could not find repo root.")


def _require_keys(obj: Dict[str, Any], keys: List[str], ctx: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise KeyError(f"Missing keys in {ctx}: {missing}")


@dataclass(frozen=True)
class SceneCatalog:
    repo_root: Path
    robot_specs_dir: Path
    scenes_dir: Path

    @classmethod
    def from_repo_root(cls, repo_root: str | Path | None = None) -> "SceneCatalog":
        root = Path(repo_root).resolve() if repo_root else find_repo_root()
        if not (root / "robot_specs").exists():
            raise FileNotFoundError(f"Missing robot_specs/ under {root}")
        if not (root / "scenes").exists():
            raise FileNotFoundError(f"Missing scenes/ under {root}")
        return cls(
            repo_root=root,
            robot_specs_dir=root / "robot_specs",
            scenes_dir=root / "scenes",
        )

    def _robot_spec_path(self, robot_id: str) -> Path:
        rid = canon_robot_id(robot_id)
        p = self.robot_specs_dir / f"{rid}.yaml"
        if p.exists():
            return p
        for cand in self.robot_specs_dir.glob("*.yaml"):
            if canon_robot_id(cand.stem) == rid:
                return cand
        raise FileNotFoundError(f"Robot spec not found: {robot_id!r} -> {rid!r}")

    @lru_cache(maxsize=None)
    def _robot_spec_cached(self, robot_id: str) -> Dict[str, Any]:
        return _load_yaml(self._robot_spec_path(robot_id))

    def load_robot_spec(self, robot_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._robot_spec_cached(robot_id))

    def robot_rmax_m(self, robot_id: str) -> float:
        spec = self._robot_spec_cached(robot_id)
        rmax = float(spec["dynamics"]["Rmax"])
        if not (rmax > 0):
            raise ValueError(f"Invalid Rmax={rmax} for {robot_id!r}")
        return rmax

    def robot_name(self, robot_id: str) -> str:
        spec = self._robot_spec_cached(robot_id)
        return str(spec.get("name", canon_robot_id(robot_id)))

    def _scene_template_path(self, scene_id: str) -> Path:
        """Find scene YAML by canonical ID. Tries exact match, then case-insensitive."""
        sid = canon_scene_id(scene_id)

        # Try direct filename mappings
        candidates = [
            self.scenes_dir / f"{sid}.yaml",
            self.scenes_dir / f"{sid.replace('_', ' ').title().replace(' ', '_')}.yaml",
        ]

        # Special mappings for known scenes
        name_map = {
            "open_reach": "Open_Reach.yaml",
            "shelf_reach": "Shelf_Reach.yaml",
        }
        if sid in name_map:
            candidates.insert(0, self.scenes_dir / name_map[sid])

        for p in candidates:
            if p.exists():
                return p

        # Fallback: scan directory
        for cand in self.scenes_dir.glob("*.yaml"):
            if canon_scene_id(cand.stem) == sid:
                return cand

        raise FileNotFoundError(
            f"Scene template not found for scene_id={scene_id!r} "
            f"(canonical={sid!r}). Checked: {[str(c) for c in candidates]}"
        )

    @lru_cache(maxsize=None)
    def _scene_template_cached(self, scene_id: str) -> Dict[str, Any]:
        sid = canon_scene_id(scene_id)
        p = self._scene_template_path(sid)
        tmpl = _load_yaml(p)
        tmpl.setdefault("schema_version", "2.0")
        tmpl["scene_id"] = sid
        tmpl["_template_path"] = str(p)
        return tmpl

    def load_scene_template(self, scene_id: str) -> Dict[str, Any]:
        return copy.deepcopy(self._scene_template_cached(scene_id))

    def build_scaled_scene(self, robot_id: str, scene_id: str) -> Dict[str, Any]:
        rid = canon_robot_id(robot_id)
        sid = canon_scene_id(scene_id)
        tmpl = self._scene_template_cached(sid)
        rmax = self.robot_rmax_m(rid)
        rname = self.robot_name(rid)

        base = {
            "schema_version": str(tmpl.get("schema_version", "2.0")),
            "scene_id": sid,
            "name": tmpl.get("name", sid),
            "description": str(tmpl.get("description", "")),
            "robot_id": rid,
            "robot_name": rname,
            "Rmax_m": rmax,
            "normalized": False,
            "source_template": str(tmpl.get("_template_path", "")),
        }

        if sid == "open_reach":
            fracs = tmpl.get("Rmax_fraction_samples") or tmpl.get("radius_fractions")
            if fracs is None:
                raise KeyError("Open_Reach missing Rmax_fraction_samples")
            fracs_f = [float(f) for f in fracs]

            base["sampling"] = {
                "radius_fractions": fracs_f,
                "radii_m": [f * rmax for f in fracs_f],
                "azimuth_range_deg": [float(x) for x in tmpl["azimuth_range_deg"]],
                "elevation_range_deg": [float(x) for x in tmpl["elevation_range_deg"]],
                "num_points": int(tmpl["num_points"]),
            }
            return copy.deepcopy(base)

        if sid == "shelf_reach":
            geom = tmpl.get("geometry", {})
            if not isinstance(geom, dict):
                raise KeyError("Shelf_Reach missing geometry mapping")

            width_f = float(geom.get("width_fraction", 0.25))
            height_f = float(geom.get("height_fraction", 0.2))
            depth_f = float(geom.get("depth_fraction", 0.9))
            z0_f = float(tmpl.get("vertical_offset_fraction", 0.15))

            cone_deg = tmpl.get("approach_cone_deg", geom.get("approach_cone_deg"))
            if cone_deg is None:
                raise KeyError("Shelf_Reach missing approach_cone_deg")
            cone_deg = float(cone_deg)

            base["geometry"] = {
                "fractions_of_Rmax": {
                    "width_fraction": width_f,
                    "height_fraction": height_f,
                    "depth_fraction": depth_f,
                },
                "geometry_m": {
                    "width_m": width_f * rmax,
                    "height_m": height_f * rmax,
                    "depth_m": depth_f * rmax,
                },
                "vertical_offset_fraction": z0_f,
                "vertical_offset_m": z0_f * rmax,
                "approach_cone_deg": cone_deg,
            }
            base["sampling"] = {"num_points": int(tmpl.get("num_points", 1200))}
            return copy.deepcopy(base)

        raise KeyError(f"No scaling rule for scene_id={sid!r}")

    def available_scenes(self) -> List[str]:
        """Discover available scene IDs from the scenes directory."""
        scenes = []
        for p in sorted(self.scenes_dir.glob("*.yaml")):
            sid = canon_scene_id(p.stem)
            scenes.append(sid)
        return scenes

    def validate(self) -> None:
        """Validate all robot specs and scene templates."""
        robot_ids = sorted({
            canon_robot_id(p.stem) for p in self.robot_specs_dir.glob("*.yaml")
        })
        if not robot_ids:
            raise FileNotFoundError(f"No robot specs in {self.robot_specs_dir}")

        for rid in robot_ids:
            _ = self.robot_rmax_m(rid)

        # Validate all discovered scenes
        for sid in self.available_scenes():
            tmpl = self._scene_template_cached(sid)
            # Build scaled scene for each robot to verify
            for rid in robot_ids:
                try:
                    _ = self.build_scaled_scene(rid, sid)
                except Exception as e:
                    print(f"WARNING: build_scaled_scene({rid!r}, {sid!r}) failed: {e}")
