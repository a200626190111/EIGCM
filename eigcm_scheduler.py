#!/usr/bin/env python3
"""Configuration-driven workflow selector and scheduler for EIGCM.

The scheduler has three deliberately separate operations:

* ``inspect`` parses Radiance materials and XML BSDF files;
* ``plan`` builds the selected calculation DAG without running Radiance;
* ``run`` executes the DAG with resumable, file-based caching.

It uses only the Python standard library so it can run in the same WSL or
Linux environment as Radiance without installing another package.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


VERSION = "0.2.0"

MATERIAL_TYPES = {
    "plastic",
    "metal",
    "trans",
    "trans2",
    "glass",
    "dielectric",
    "interface",
    "mirror",
    "prism1",
    "prism2",
    "brtdfunc",
    "bsdf",
    "absdf",
    "ashik2",
    "illum",
    "glow",
    "light",
    "spotlight",
    "mist",
    "antimatter",
    "plasfunc",
    "metfunc",
    "transfunc",
    "plasdata",
    "metdata",
    "transdata",
    "brightfunc",
    "brightdata",
    "brighttext",
    "colortext",
    "colorfunc",
    "colordata",
    "mixfunc",
    "mixdata",
    "mixtext",
    "mixpict",
    "texfunc",
    "texdata",
    "alias",
}

GEOMETRY_TYPES = {
    "polygon",
    "sphere",
    "bubble",
    "ring",
    "cylinder",
    "tube",
    "cone",
    "cup",
    "source",
    "instance",
    "mesh",
}

PATTERN_TYPES = {
    "brightfunc",
    "brightdata",
    "brighttext",
    "colortext",
    "colorfunc",
    "colordata",
    "mixfunc",
    "mixdata",
    "mixtext",
    "mixpict",
    "texfunc",
    "texdata",
}

ALL_PRIMITIVE_TYPES = MATERIAL_TYPES | GEOMETRY_TYPES | PATTERN_TYPES

ANALYTIC_TRANSMISSION_TYPES = {
    "glass",
    "dielectric",
    "interface",
    "trans",
    "trans2",
    "prism1",
    "prism2",
    "brtdfunc",
}

BSDF_TYPES = {"bsdf", "absdf"}

QUALITY_PROFILES: Mapping[str, Mapping[str, object]] = {
    "draft": {
        "background_ad": 10000,
        "background_lw": 0.0001,
        "view_ad": 10000,
        "view_lw": 0.0001,
        "daylight_ad": 500,
        "daylight_lw": 0.002,
        "daylight_c": 64,
        "sun_samples": 1000,
        "lobe_samples": 8,
        "lobe_disk_resolution": 4,
        "bsdf_samples": 64,
        "bsdf_matrix_samples": 256,
        "sun_disk_samples": 16,
        "secondary_sun_disk_samples": 4,
        "rough_samples": 256,
        "rough_secondary_samples": 4,
        "reflection_level": 4,
    },
    "balanced": {
        "background_ad": 50000,
        "background_lw": 0.00002,
        "view_ad": 40000,
        "view_lw": 0.000025,
        "daylight_ad": 1000,
        "daylight_lw": 0.001,
        "daylight_c": 250,
        "sun_samples": 10000,
        "lobe_samples": 16,
        "lobe_disk_resolution": 8,
        "bsdf_samples": 256,
        "bsdf_matrix_samples": 1024,
        "sun_disk_samples": 64,
        "secondary_sun_disk_samples": 16,
        "rough_samples": 1024,
        "rough_secondary_samples": 16,
        "reflection_level": 5,
    },
    "high": {
        "background_ad": 100000,
        "background_lw": 0.00001,
        "view_ad": 80000,
        "view_lw": 0.0000125,
        "daylight_ad": 4000,
        "daylight_lw": 0.00025,
        "daylight_c": 1000,
        "sun_samples": 40000,
        "lobe_samples": 64,
        "lobe_disk_resolution": 16,
        "bsdf_samples": 1024,
        "bsdf_matrix_samples": 4096,
        "sun_disk_samples": 256,
        "secondary_sun_disk_samples": 64,
        "rough_samples": 4096,
        "rough_secondary_samples": 64,
        "reflection_level": 6,
    },
}


class SchedulerError(RuntimeError):
    """A user-facing configuration or execution error."""


@dataclass
class Primitive:
    modifier: str
    kind: str
    name: str
    string_args: List[str]
    integer_args: List[int]
    real_args: List[float]
    source: str


@dataclass
class BsdfInfo:
    path: str
    representation: str
    incident_data_structure: str = ""
    angle_bases: List[str] = field(default_factory=list)
    directions: List[str] = field(default_factory=list)
    readable: bool = True
    error: str = ""


@dataclass
class FenestrationSurface:
    name: str
    material: str
    material_kind: str
    source: str
    vertices: List[Tuple[float, float, float]]
    normal: Tuple[float, float, float]
    centroid: Tuple[float, float, float]
    area: float
    xml: str = ""


@dataclass
class DetectionResult:
    files_scanned: List[str]
    materials: List[Primitive]
    active_material_names: List[str]
    analytic_transmission_materials: List[str]
    specular_materials: List[str]
    rough_specular_materials: List[str]
    bsdf_materials: List[str]
    absdf_materials: List[str]
    bsdf_files: List[BsdfInfo]
    fenestration_surfaces: List[FenestrationSurface]
    normal_reference: str
    normal_reference_count: int
    warnings: List[str]

    @property
    def has_analytic_transmission(self) -> bool:
        return bool(self.analytic_transmission_materials)

    @property
    def has_specular(self) -> bool:
        return bool(self.specular_materials)

    @property
    def has_bsdf(self) -> bool:
        return bool(self.bsdf_materials or self.absdf_materials or self.bsdf_files)

    @property
    def has_absdf(self) -> bool:
        return bool(self.absdf_materials)

    @property
    def has_tensortree(self) -> bool:
        return any(item.representation == "TensorTree" for item in self.bsdf_files)

    def to_dict(self) -> Dict[str, object]:
        return {
            "files_scanned": self.files_scanned,
            "active_material_names": self.active_material_names,
            "analytic_transmission_materials": self.analytic_transmission_materials,
            "specular_materials": self.specular_materials,
            "rough_specular_materials": self.rough_specular_materials,
            "bsdf_materials": self.bsdf_materials,
            "absdf_materials": self.absdf_materials,
            "bsdf_files": [dataclasses.asdict(item) for item in self.bsdf_files],
            "fenestration_surfaces": [
                {
                    "name": item.name,
                    "material": item.material,
                    "material_kind": item.material_kind,
                    "source": item.source,
                    "normal": item.normal,
                    "centroid": item.centroid,
                    "area": item.area,
                    "xml": item.xml,
                }
                for item in self.fenestration_surfaces
            ],
            "normal_reference": self.normal_reference,
            "normal_reference_count": self.normal_reference_count,
            "features": {
                "analytic_transmission": self.has_analytic_transmission,
                "specular_reflection": self.has_specular,
                "bsdf": self.has_bsdf,
                "absdf": self.has_absdf,
                "tensortree": self.has_tensortree,
            },
            "warnings": self.warnings,
        }


@dataclass
class WorkflowDecision:
    representation: str
    ev_strategy: str
    directlobecontrast: bool
    specularcontrast: bool
    bsdf_matrix: bool
    ttsuncontrast: bool
    reasons: Dict[str, str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class Step:
    step_id: str
    stage: str
    description: str
    command: str
    outputs: List[str]
    inputs: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


@dataclass
class WindowGroup:
    name: str
    receiver: str
    mode: str
    xml: str = ""
    open_scene: List[str] = field(default_factory=list)
    open_materials: List[str] = field(default_factory=list)
    open_black_materials: List[str] = field(default_factory=list)
    tds_total: str = ""
    tds_direct: str = ""
    auto_detected: bool = False
    materials: List[str] = field(default_factory=list)
    surface_names: List[str] = field(default_factory=list)
    inward_normal: List[float] = field(default_factory=list)
    receiver_content: str = ""


@dataclass
class Plan:
    config_path: str
    config_hash: str
    output_dir: str
    execution_dir: str
    path_prepend: str
    nview: int
    quality: str
    detection: DetectionResult
    decision: WorkflowDecision
    groups: List[WindowGroup]
    steps: List[Step]
    warnings: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "scheduler_version": VERSION,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "output_dir": self.output_dir,
            "execution_dir": self.execution_dir,
            "path_prepend": self.path_prepend,
            "nview": self.nview,
            "quality": self.quality,
            "detection": self.detection.to_dict(),
            "decision": self.decision.to_dict(),
            "window_groups": [dataclasses.asdict(group) for group in self.groups],
            "warnings": self.warnings,
            "steps": [step.to_dict() for step in self.steps],
        }


class Settings:
    def __init__(self, path: str, overrides: Sequence[str] = ()) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise SchedulerError("Configuration file does not exist: {}".format(self.path))
        self.base_dir = self.path.parent
        self.parser = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(),
            inline_comment_prefixes=("#", ";"),
        )
        self.parser.optionxform = str.lower
        with self.path.open("r", encoding="utf-8-sig") as handle:
            self.parser.read_file(handle)
        self._apply_overrides(overrides)
        self.quality = self.get("workflow", "quality", "balanced").lower()
        if self.quality not in QUALITY_PROFILES:
            raise SchedulerError(
                "Unknown workflow.quality {!r}; choose draft, balanced, or high".format(
                    self.quality
                )
            )

    def _apply_overrides(self, overrides: Sequence[str]) -> None:
        for override in overrides:
            match = re.fullmatch(r"([^.]+)\.([^=]+)=(.*)", override)
            if not match:
                raise SchedulerError(
                    "Invalid --set {!r}; use section.option=value".format(override)
                )
            section, option, value = match.groups()
            if not self.parser.has_section(section):
                self.parser.add_section(section)
            self.parser.set(section, option, value)

    def get(self, section: str, option: str, fallback: str = "") -> str:
        return self.parser.get(section, option, fallback=fallback).strip()

    def getint(self, section: str, option: str, fallback: int) -> int:
        value = self.get(section, option, "")
        if value:
            try:
                return int(value)
            except ValueError as exc:
                raise SchedulerError(
                    "{}.{} must be an integer, got {!r}".format(section, option, value)
                ) from exc
        profile_value = QUALITY_PROFILES[self.quality].get(option, fallback)
        return int(profile_value)

    def getfloat(self, section: str, option: str, fallback: float) -> float:
        value = self.get(section, option, "")
        if value:
            try:
                return float(value)
            except ValueError as exc:
                raise SchedulerError(
                    "{}.{} must be numeric, got {!r}".format(section, option, value)
                ) from exc
        profile_value = QUALITY_PROFILES[self.quality].get(option, fallback)
        return float(profile_value)

    def getbool(self, section: str, option: str, fallback: bool) -> bool:
        value = self.get(section, option, "")
        if not value:
            return fallback
        lowered = value.lower()
        if lowered in {"1", "yes", "true", "on"}:
            return True
        if lowered in {"0", "no", "false", "off"}:
            return False
        raise SchedulerError(
            "{}.{} must be true or false, got {!r}".format(section, option, value)
        )

    def getlist(self, section: str, option: str) -> List[str]:
        return split_config_list(self.get(section, option, ""))

    def path_value(self, section: str, option: str, fallback: str = "") -> str:
        value = self.get(section, option, fallback)
        return resolve_config_path(value, self.base_dir) if value else ""

    def path_list(self, section: str, option: str) -> List[str]:
        return [resolve_config_path(value, self.base_dir) for value in self.getlist(section, option)]

    def command(self, name: str, fallback: Optional[str] = None) -> str:
        return self.get("commands", name, fallback or name)

    def hash(self) -> str:
        text = self.path.read_bytes()
        return hashlib.sha256(text).hexdigest()


def split_config_list(value: str) -> List[str]:
    if not value.strip():
        return []
    if "\n" in value:
        result: List[str] = []
        for line in value.splitlines():
            stripped = line.strip().rstrip(",")
            if stripped:
                result.extend(shlex.split(stripped, posix=True))
        return result
    return shlex.split(value.replace(",", " "), posix=True)


def resolve_config_path(value: str, base_dir: Path) -> str:
    expanded = os.path.expandvars(os.path.expanduser(value.strip()))
    if not expanded:
        return ""
    if expanded.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", expanded):
        return expanded
    return str((base_dir / expanded).resolve())


def host_path(value: str) -> Path:
    """Map a WSL /mnt/<drive> path to Windows for read-only inspection."""
    if os.name == "nt":
        match = re.match(r"^/mnt/([A-Za-z])/(.*)$", value)
        if match:
            drive, tail = match.groups()
            return Path("{}:\\{}".format(drive.upper(), tail.replace("/", "\\")))
    return Path(value)


def q(value: object) -> str:
    return shlex.quote(str(value))


def join_args(parts: Iterable[object]) -> str:
    return " ".join(q(part) for part in parts if str(part) != "")


def clean_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def strip_rad_commands(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines)


def tokenize_radiance(text: str) -> List[str]:
    lexer = shlex.shlex(strip_rad_commands(text), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def parse_radiance_file(path: Path) -> Tuple[List[Primitive], List[str]]:
    warnings: List[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], ["Could not read {}: {}".format(path, exc)]
    tokens = tokenize_radiance(text)
    primitives: List[Primitive] = []
    index = 0
    while index + 2 < len(tokens):
        modifier, kind, name = tokens[index : index + 3]
        kind_lower = kind.lower()
        if kind_lower not in ALL_PRIMITIVE_TYPES:
            index += 1
            continue
        index += 3
        if kind_lower == "alias":
            if index < len(tokens):
                primitives.append(
                    Primitive(modifier, kind_lower, name, [tokens[index]], [], [], str(path))
                )
                index += 1
            continue
        argument_groups: List[List[str]] = []
        valid = True
        for _ in range(3):
            if index >= len(tokens):
                valid = False
                break
            try:
                count = int(tokens[index])
            except ValueError:
                valid = False
                break
            index += 1
            if count < 0 or index + count > len(tokens):
                valid = False
                break
            argument_groups.append(tokens[index : index + count])
            index += count
        if not valid or len(argument_groups) != 3:
            warnings.append(
                "Could not parse Radiance primitive {} {} in {}".format(kind, name, path)
            )
            continue
        integer_args: List[int] = []
        real_args: List[float] = []
        try:
            integer_args = [int(value) for value in argument_groups[1]]
        except ValueError:
            warnings.append("Non-integer integer argument in {} ({})".format(name, path))
        try:
            real_args = [float(value) for value in argument_groups[2]]
        except ValueError:
            warnings.append("Non-numeric real argument in {} ({})".format(name, path))
        primitives.append(
            Primitive(
                modifier=modifier,
                kind=kind_lower,
                name=name,
                string_args=argument_groups[0],
                integer_args=integer_args,
                real_args=real_args,
                source=str(path),
            )
        )
    return primitives, warnings


def rad_includes(path: Path) -> List[Path]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    included: List[Path] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("!"):
            continue
        try:
            parts = shlex.split(stripped[1:], posix=True)
        except ValueError:
            continue
        for part in parts[1:]:
            if part.startswith("-"):
                continue
            if not re.search(r"\.(?:rad|mat)$", part, re.I):
                continue
            candidate = Path(part)
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if candidate.exists():
                included.append(candidate.resolve())
    return included


def discover_rad_files(paths: Sequence[str]) -> Tuple[List[Path], List[str]]:
    queue = [host_path(path) for path in paths]
    visited: Set[str] = set()
    discovered: List[Path] = []
    warnings: List[str] = []
    while queue:
        candidate = queue.pop(0)
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in visited:
            continue
        visited.add(key)
        if not candidate.exists():
            warnings.append("Radiance input was not available for inspection: {}".format(candidate))
            continue
        discovered.append(candidate)
        queue.extend(rad_includes(candidate))
    return discovered, warnings


def active_materials(primitives: Sequence[Primitive]) -> List[Primitive]:
    by_name = {p.name: p for p in primitives if p.kind in MATERIAL_TYPES}
    used = {
        p.modifier
        for p in primitives
        if p.kind in GEOMETRY_TYPES and p.modifier not in {"", "void"}
    }
    if not used:
        return list(by_name.values())
    active_names: Set[str] = set()
    pending = list(used)
    while pending:
        name = pending.pop()
        if name in active_names:
            continue
        active_names.add(name)
        primitive = by_name.get(name)
        if primitive:
            if primitive.modifier not in {"", "void"}:
                pending.append(primitive.modifier)
            if primitive.kind == "alias" and primitive.string_args:
                pending.append(primitive.string_args[0])
    selected = [p for name, p in by_name.items() if name in active_names]
    return selected or list(by_name.values())


def xml_path_from_primitive(primitive: Primitive) -> str:
    for value in primitive.string_args:
        if value.lower().endswith(".xml"):
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = Path(primitive.source).parent / candidate
            return str(candidate.resolve())
    return ""


def command_path_from_host(path_value: str, prefer_wsl: bool) -> str:
    """Convert an inspected Windows path back to /mnt form when configs use WSL."""
    if prefer_wsl:
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", path_value)
        if match:
            drive, tail = match.groups()
            return "/mnt/{}/{}".format(drive.lower(), tail.replace("\\", "/"))
    return path_value


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def inspect_bsdf_xml(path_value: str) -> BsdfInfo:
    path = host_path(path_value)
    info = BsdfInfo(path=path_value, representation="Unknown")
    if not path.exists():
        info.readable = False
        info.error = "file not found during inspection"
        return info
    incident = ""
    bases: Set[str] = set()
    directions: Set[str] = set()
    try:
        for _event, element in ET.iterparse(str(path), events=("end",)):
            name = local_name(element.tag)
            value = (element.text or "").strip()
            if name == "IncidentDataStructure" and value:
                incident = value
            elif name == "AngleBasis" and value:
                bases.add(value)
            elif name == "WavelengthDataDirection" and value:
                directions.add(value)
            element.clear()
            if incident and len(bases) >= 2 and len(directions) >= 4:
                break
    except (ET.ParseError, OSError) as exc:
        info.readable = False
        info.error = str(exc)
        return info
    info.incident_data_structure = incident
    info.angle_bases = sorted(bases)
    info.directions = sorted(directions)
    combined = " ".join([incident] + sorted(bases)).lower()
    if "tensortree" in combined or "shirley" in combined:
        info.representation = "TensorTree"
    elif "klems" in combined or "columns" in combined:
        info.representation = "Klems"
    else:
        info.representation = "Unknown"
    return info


def material_specularity(primitive: Primitive) -> Tuple[float, float]:
    values = primitive.real_args
    if primitive.kind in {"plastic", "metal"} and len(values) >= 5:
        return values[3], values[4]
    if primitive.kind in {"trans", "trans2"} and len(values) >= 7:
        reflected = max(0.0, values[3])
        transmitted = max(0.0, values[5] * values[6])
        return max(reflected, transmitted), values[4]
    if primitive.kind in {"glass", "dielectric", "interface", "mirror", "prism1", "prism2"}:
        return 1.0, 0.0
    return 0.0, 0.0


def vector_subtract(
    left: Tuple[float, float, float], right: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def vector_dot(
    left: Tuple[float, float, float], right: Tuple[float, float, float]
) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def vector_length(value: Tuple[float, float, float]) -> float:
    return math.sqrt(vector_dot(value, value))


def normalized(
    value: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    length = vector_length(value)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (value[0] / length, value[1] / length, value[2] / length)


def polygon_properties(
    values: Sequence[float],
) -> Optional[
    Tuple[
        List[Tuple[float, float, float]],
        Tuple[float, float, float],
        Tuple[float, float, float],
        float,
    ]
]:
    if len(values) < 9 or len(values) % 3:
        return None
    vertices = [
        (float(values[index]), float(values[index + 1]), float(values[index + 2]))
        for index in range(0, len(values), 3)
    ]
    nx = ny = nz = 0.0
    for index, current in enumerate(vertices):
        following = vertices[(index + 1) % len(vertices)]
        nx += (current[1] - following[1]) * (current[2] + following[2])
        ny += (current[2] - following[2]) * (current[0] + following[0])
        nz += (current[0] - following[0]) * (current[1] + following[1])
    vector = (nx, ny, nz)
    area = 0.5 * vector_length(vector)
    normal = normalized(vector)
    centroid = (
        sum(point[0] for point in vertices) / len(vertices),
        sum(point[1] for point in vertices) / len(vertices),
        sum(point[2] for point in vertices) / len(vertices),
    )
    if area <= 1e-12:
        return None
    return vertices, normal, centroid, area


def resolve_material(
    modifier: str, by_name: Mapping[str, Primitive]
) -> Optional[Primitive]:
    visited: Set[str] = set()
    current = modifier
    while current not in visited and current not in {"", "void"}:
        visited.add(current)
        primitive = by_name.get(current)
        if primitive is None:
            return None
        if primitive.kind == "alias" and primitive.string_args:
            current = primitive.string_args[0]
            continue
        if primitive.kind in MATERIAL_TYPES and primitive.kind not in PATTERN_TYPES:
            return primitive
        current = primitive.modifier
    return None


def scene_geometry_center(primitives: Sequence[Primitive]) -> Tuple[float, float, float]:
    coordinates: List[Tuple[float, float, float]] = []
    for primitive in primitives:
        if primitive.kind != "polygon":
            continue
        properties = polygon_properties(primitive.real_args)
        if properties:
            coordinates.extend(properties[0])
    if not coordinates:
        return (0.0, 0.0, 0.0)
    minimum = [min(point[axis] for point in coordinates) for axis in range(3)]
    maximum = [max(point[axis] for point in coordinates) for axis in range(3)]
    return tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))  # type: ignore


def read_view_positions(
    path_value: str,
) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    path = host_path(path_value)
    warnings: List[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [], ["Could not read viewpoints from {}: {}".format(path, exc)]

    positions: List[Tuple[float, float, float]] = []
    malformed = 0
    for raw in lines:
        content = raw.split("#", 1)[0].strip()
        if not content:
            continue
        try:
            tokens = shlex.split(content, posix=True)
        except ValueError:
            malformed += 1
            continue
        coordinate_tokens: Sequence[str]
        if "-vp" in tokens:
            start = tokens.index("-vp") + 1
            coordinate_tokens = tokens[start : start + 3]
        else:
            coordinate_tokens = tokens[:3]
        if len(coordinate_tokens) != 3:
            malformed += 1
            continue
        try:
            point = tuple(float(value) for value in coordinate_tokens)
        except ValueError:
            malformed += 1
            continue
        if not all(math.isfinite(value) for value in point):
            malformed += 1
            continue
        positions.append(point)  # type: ignore[arg-type]

    if malformed:
        warnings.append(
            "Ignored {} malformed viewpoint line{} in {} while orienting window normals.".format(
                malformed, "" if malformed == 1 else "s", path
            )
        )
    if not positions:
        warnings.append(
            "No valid viewpoint positions were found in {}; using the scene geometry centre for window normals.".format(
                path
            )
        )
    return positions, warnings


def detect_fenestration_surfaces(
    primitives: Sequence[Primitive],
    prefer_wsl: bool,
    interior_point: Optional[Tuple[float, float, float]] = None,
    view_positions: Sequence[Tuple[float, float, float]] = (),
) -> List[FenestrationSurface]:
    by_name = {item.name: item for item in primitives if item.kind in MATERIAL_TYPES}
    geometry_center = scene_geometry_center(primitives)
    surfaces: List[FenestrationSurface] = []
    for primitive in primitives:
        if primitive.kind != "polygon":
            continue
        material = resolve_material(primitive.modifier, by_name)
        if material is None or material.kind not in ANALYTIC_TRANSMISSION_TYPES | BSDF_TYPES:
            continue
        properties = polygon_properties(primitive.real_args)
        if not properties:
            continue
        vertices, normal, centroid, area = properties
        if interior_point is not None:
            target = interior_point
        elif view_positions:
            target = min(
                view_positions,
                key=lambda point: vector_dot(
                    vector_subtract(point, centroid), vector_subtract(point, centroid)
                ),
            )
        else:
            target = geometry_center
        toward_interior = vector_subtract(target, centroid)
        if vector_dot(normal, toward_interior) < 0:
            vertices = list(reversed(vertices))
            normal = (-normal[0], -normal[1], -normal[2])
        xml = xml_path_from_primitive(material)
        if xml:
            xml = command_path_from_host(xml, prefer_wsl)
        surfaces.append(
            FenestrationSurface(
                name=primitive.name,
                material=material.name,
                material_kind=material.kind,
                source=primitive.source,
                vertices=vertices,
                normal=normal,
                centroid=centroid,
                area=area,
                xml=xml,
            )
        )
    return surfaces


def inspect_scene(settings: Settings) -> DetectionResult:
    configured = settings.path_list("scene", "materials")
    configured += settings.path_list("scene", "scene")
    configured += settings.path_list("scene", "additional_inputs")
    files, warnings = discover_rad_files(configured)
    primitives: List[Primitive] = []
    for path in files:
        parsed, parse_warnings = parse_radiance_file(path)
        primitives.extend(parsed)
        warnings.extend(parse_warnings)
    active = active_materials(primitives)

    include = set(settings.getlist("workflow", "material_include"))
    exclude = set(settings.getlist("workflow", "material_exclude"))
    if include:
        active = [item for item in active if item.name in include]
    if exclude:
        active = [item for item in active if item.name not in exclude]

    specularity_threshold = settings.getfloat(
        "workflow", "specularity_threshold", 0.05
    )
    roughness_threshold = settings.getfloat("workflow", "roughness_threshold", 1e-4)

    analytic = sorted(
        {item.name for item in active if item.kind in ANALYTIC_TRANSMISSION_TYPES}
    )
    specular: Set[str] = set()
    rough: Set[str] = set()
    for item in active:
        strength, roughness = material_specularity(item)
        if strength >= specularity_threshold:
            specular.add(item.name)
            if roughness > roughness_threshold:
                rough.add(item.name)

    bsdf_items = [item for item in active if item.kind in BSDF_TYPES]
    bsdf_names = sorted({item.name for item in bsdf_items if item.kind == "bsdf"})
    absdf_names = sorted({item.name for item in bsdf_items if item.kind == "absdf"})
    prefer_wsl = any(value.startswith("/mnt/") for value in configured)
    interior_values = settings.getlist("auto_window_groups", "interior_point")
    interior_point: Optional[Tuple[float, float, float]] = None
    if interior_values:
        if len(interior_values) != 3:
            raise SchedulerError("auto_window_groups.interior_point must contain x y z")
        try:
            interior_point = tuple(float(value) for value in interior_values)  # type: ignore
        except ValueError as exc:
            raise SchedulerError(
                "auto_window_groups.interior_point must contain three numbers"
            ) from exc
    view_positions: List[Tuple[float, float, float]] = []
    if interior_point is not None:
        normal_reference = "configured interior_point"
        normal_reference_count = 1
    else:
        views_path = settings.path_value("scene", "views")
        if views_path:
            view_positions, view_warnings = read_view_positions(views_path)
            warnings.extend(view_warnings)
        if view_positions:
            normal_reference = "nearest viewpoint"
            normal_reference_count = len(view_positions)
        else:
            normal_reference = "scene geometry centre"
            normal_reference_count = 1
    fenestration_surfaces = detect_fenestration_surfaces(
        primitives, prefer_wsl, interior_point, view_positions
    )
    xml_paths: List[str] = []
    for item in bsdf_items:
        value = xml_path_from_primitive(item)
        value = command_path_from_host(value, prefer_wsl) if value else ""
        if value and value not in xml_paths:
            xml_paths.append(value)
    for value in settings.path_list("bsdf", "xml"):
        if value not in xml_paths:
            xml_paths.append(value)
    for section in settings.parser.sections():
        lowered = section.lower()
        if lowered.startswith("windowgroup:") or lowered.startswith("windowgroup "):
            value = settings.path_value(section, "xml")
            if value and value not in xml_paths:
                xml_paths.append(value)
    bsdf_files = [inspect_bsdf_xml(value) for value in xml_paths]
    for item in bsdf_files:
        if not item.readable:
            warnings.append("Could not inspect BSDF XML {}: {}".format(item.path, item.error))

    return DetectionResult(
        files_scanned=[str(path) for path in files],
        materials=active,
        active_material_names=sorted({item.name for item in active}),
        analytic_transmission_materials=analytic,
        specular_materials=sorted(specular),
        rough_specular_materials=sorted(rough),
        bsdf_materials=bsdf_names,
        absdf_materials=absdf_names,
        bsdf_files=bsdf_files,
        fenestration_surfaces=fenestration_surfaces,
        normal_reference=normal_reference,
        normal_reference_count=normal_reference_count,
        warnings=warnings,
    )


def tri_state(settings: Settings, option: str, automatic: bool) -> Tuple[bool, str]:
    value = settings.get("workflow", option, "auto").lower()
    if value == "auto":
        return automatic, "automatically {}".format("enabled" if automatic else "disabled")
    if value in {"1", "yes", "true", "on", "enable", "enabled"}:
        return True, "forced on by workflow.{}".format(option)
    if value in {"0", "no", "false", "off", "disable", "disabled"}:
        return False, "forced off by workflow.{}".format(option)
    raise SchedulerError(
        "workflow.{} must be auto, true, or false; got {!r}".format(option, value)
    )


def auto_bool(
    settings: Settings, section: str, option: str, automatic: bool
) -> Tuple[bool, str]:
    value = settings.get(section, option, "auto").lower()
    if value == "auto":
        return automatic, "automatically {}".format("enabled" if automatic else "disabled")
    if value in {"1", "yes", "true", "on", "enable", "enabled"}:
        return True, "forced on"
    if value in {"0", "no", "false", "off", "disable", "disabled"}:
        return False, "forced off"
    raise SchedulerError(
        "{}.{} must be auto, true, or false; got {!r}".format(section, option, value)
    )


def orientation_name(inward: Tuple[float, float, float]) -> str:
    outward = (-inward[0], -inward[1], -inward[2])
    if outward[2] >= 0.7:
        return "skylight"
    if outward[2] <= -0.7:
        return "downward"
    angle = math.degrees(math.atan2(outward[0], outward[1])) % 360.0
    names = [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]
    return names[int((angle + 22.5) // 45.0) % 8]


def weighted_group_normal(surfaces: Sequence[FenestrationSurface]) -> Tuple[float, float, float]:
    vector = (
        sum(item.normal[0] * item.area for item in surfaces),
        sum(item.normal[1] * item.area for item in surfaces),
        sum(item.normal[2] * item.area for item in surfaces),
    )
    return normalized(vector)


def receiver_content(
    name: str,
    surfaces: Sequence[FenestrationSurface],
    normal: Tuple[float, float, float],
    offset: float,
    mf: int,
) -> str:
    modifier = "eigcm_receiver_{}".format(clean_id(name))
    up_axis = "Z" if abs(normal[2]) < 0.9 else "Y"
    lines = [
        "# Automatically generated by eigcm_scheduler.py",
        "#@rfluxmtx h=r{} u={}".format(mf, up_axis),
        "void glow {}".format(modifier),
        "0",
        "0",
        "4 1 1 1 0",
        "",
    ]
    for index, surface in enumerate(surfaces, start=1):
        shifted = [
            (
                point[0] + normal[0] * offset,
                point[1] + normal[1] * offset,
                point[2] + normal[2] * offset,
            )
            for point in surface.vertices
        ]
        lines.extend(
            [
                "{} polygon auto_{}_{}".format(modifier, clean_id(surface.name), index),
                "0",
                "0",
                str(3 * len(shifted)),
            ]
        )
        lines.extend(
            "    {:.9g} {:.9g} {:.9g}".format(point[0], point[1], point[2])
            for point in shifted
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def infer_open_scene(scene_paths: Sequence[str]) -> List[str]:
    for value in scene_paths:
        path = host_path(value)
        candidates = [path.with_name(path.stem + "_no_window" + path.suffix)]
        if "-" in path.stem:
            candidates.append(path.with_name(path.stem + "_no_window" + path.suffix))
        for candidate in candidates:
            if candidate.exists():
                prefer_wsl = value.startswith("/mnt/")
                return [command_path_from_host(str(candidate.resolve()), prefer_wsl)]
    return []


def discover_window_groups(
    settings: Settings, detection: DetectionResult
) -> List[WindowGroup]:
    include = set(settings.getlist("auto_window_groups", "material_include"))
    exclude = set(settings.getlist("auto_window_groups", "material_exclude"))
    minimum_area = settings.getfloat("auto_window_groups", "minimum_area", 0.01)
    tolerance = settings.getfloat(
        "auto_window_groups", "orientation_tolerance_deg", 5.0
    )
    offset = settings.getfloat("auto_window_groups", "receiver_offset", 0.05)
    if tolerance <= 0 or tolerance > 45:
        raise SchedulerError(
            "auto_window_groups.orientation_tolerance_deg must be in (0, 45]"
        )
    candidates = [
        item
        for item in detection.fenestration_surfaces
        if item.area >= minimum_area
        and (not include or item.material in include)
        and item.material not in exclude
    ]
    clusters: List[List[FenestrationSurface]] = []
    cosine = math.cos(math.radians(tolerance))
    for surface in sorted(candidates, key=lambda item: (item.material, item.name)):
        destination: Optional[List[FenestrationSurface]] = None
        for cluster in clusters:
            representative = weighted_group_normal(cluster)
            first = cluster[0]
            same_optics = (
                first.material == surface.material
                and first.material_kind == surface.material_kind
                and first.xml == surface.xml
            )
            if same_optics and vector_dot(representative, surface.normal) >= cosine:
                destination = cluster
                break
        if destination is None:
            clusters.append([surface])
        else:
            destination.append(surface)

    labels = [orientation_name(weighted_group_normal(cluster)) for cluster in clusters]
    label_counts = {label: labels.count(label) for label in set(labels)}
    out = required_path(settings, "scene", "out")
    mf = settings.getint("radiance", "mf", 1)
    global_open_scene = settings.path_list("auto_window_groups", "open_scene")
    if not global_open_scene:
        global_open_scene = infer_open_scene(settings.path_list("scene", "scene"))
    global_open_materials = settings.path_list("auto_window_groups", "open_materials")
    global_open_black = settings.path_list(
        "auto_window_groups", "open_black_materials"
    )
    groups: List[WindowGroup] = []
    used_names: Set[str] = set()
    for cluster, label in zip(clusters, labels):
        material = cluster[0].material
        base_name = label
        if label_counts[label] > 1:
            base_name = "{}_{}".format(label, clean_id(material).lower())
        name = base_name
        suffix = 2
        while name in used_names:
            name = "{}_{}".format(base_name, suffix)
            suffix += 1
        used_names.add(name)
        normal = weighted_group_normal(cluster)
        mode = "bsdf" if cluster[0].material_kind in BSDF_TYPES else "geometry"
        receiver = out.rstrip("/\\") + "/receivers/{}_mf{}.rad".format(
            clean_id(name), mf
        )
        groups.append(
            WindowGroup(
                name=name,
                receiver=receiver,
                mode=mode,
                xml=cluster[0].xml,
                open_scene=list(global_open_scene) if mode == "bsdf" else [],
                open_materials=list(global_open_materials),
                open_black_materials=list(global_open_black),
                auto_detected=True,
                materials=sorted({item.material for item in cluster}),
                surface_names=[item.name for item in cluster],
                inward_normal=[normal[0], normal[1], normal[2]],
                receiver_content=receiver_content(name, cluster, normal, offset, mf),
            )
        )
    return groups


def read_window_groups(settings: Settings, detection: DetectionResult) -> List[WindowGroup]:
    explicit_groups: List[WindowGroup] = []
    detected_xml = [item.path for item in detection.bsdf_files]
    for section in settings.parser.sections():
        lowered = section.lower()
        if not (lowered.startswith("windowgroup:") or lowered.startswith("windowgroup ")):
            continue
        name = re.split(r"[: ]", section, maxsplit=1)[1].strip()
        receiver = settings.path_value(section, "receiver")
        if not receiver:
            raise SchedulerError("{}.receiver is required".format(section))
        xml = settings.path_value(section, "xml")
        mode = settings.get(section, "mode", "auto").lower()
        if mode == "auto":
            mode = "bsdf" if xml else "geometry"
        if mode not in {"geometry", "bsdf", "precomputed"}:
            raise SchedulerError(
                "{}.mode must be auto, geometry, bsdf, or precomputed".format(section)
            )
        if mode == "bsdf" and not xml and len(detected_xml) == 1:
            xml = detected_xml[0]
        explicit_groups.append(
            WindowGroup(
                name=name,
                receiver=receiver,
                mode=mode,
                xml=xml,
                open_scene=settings.path_list(section, "open_scene"),
                open_materials=settings.path_list(section, "open_materials"),
                open_black_materials=settings.path_list(section, "open_black_materials"),
                tds_total=settings.path_value(section, "tds_total"),
                tds_direct=settings.path_value(section, "tds_direct"),
            )
        )
    enabled_value = settings.get("auto_window_groups", "enabled", "auto").lower()
    if enabled_value == "auto":
        auto_enabled = not explicit_groups
    elif enabled_value in {"1", "yes", "true", "on", "enabled"}:
        auto_enabled = True
    elif enabled_value in {"0", "no", "false", "off", "disabled"}:
        auto_enabled = False
    else:
        raise SchedulerError(
            "auto_window_groups.enabled must be auto, true, or false"
        )
    automatic = discover_window_groups(settings, detection) if auto_enabled else []
    merged = {group.name: group for group in automatic}
    for group in explicit_groups:
        merged[group.name] = group
    groups = list(merged.values())
    groups.sort(key=lambda item: item.name)
    if auto_enabled and not automatic:
        detection.warnings.append(
            "Automatic window grouping found no polygon using a transmitting or BSDF material."
        )
    return groups


def choose_workflow(
    settings: Settings, detection: DetectionResult, groups: Sequence[WindowGroup]
) -> WorkflowDecision:
    group_bsdf = any(group.mode == "bsdf" for group in groups)
    has_bsdf = detection.has_bsdf or group_bsdf
    direct, direct_reason = tri_state(
        settings, "directlobecontrast", detection.has_analytic_transmission
    )
    specular_auto = detection.has_specular or bool(
        settings.path_value("specularcontrast", "allowlist")
    )
    specular, specular_reason = tri_state(settings, "specularcontrast", specular_auto)
    bsdf_matrix, bsdf_matrix_reason = tri_state(settings, "bsdf_matrix", has_bsdf)
    ttsun_auto = has_bsdf and (detection.has_tensortree or detection.has_absdf)
    ttsun, ttsun_reason = tri_state(settings, "ttsuncontrast", ttsun_auto)

    requested_strategy = settings.get("workflow", "ev_strategy", "auto").lower()
    if requested_strategy == "auto":
        ev_strategy = "bsdf_sun" if bsdf_matrix else "path_components"
        ev_reason = "selected from the detected fenestration representation"
    elif requested_strategy in {"path_components", "bsdf_sun", "total"}:
        ev_strategy = requested_strategy
        ev_reason = "forced by workflow.ev_strategy"
    else:
        raise SchedulerError(
            "workflow.ev_strategy must be auto, path_components, bsdf_sun, or total"
        )

    if has_bsdf and detection.has_analytic_transmission:
        representation = "mixed geometry and BSDF"
    elif has_bsdf:
        representation = "TensorTree BSDF" if detection.has_tensortree else "BSDF"
    else:
        representation = "explicit geometry"

    warnings: List[str] = []
    if ttsun and not bsdf_matrix:
        warnings.append(
            "ttsuncontrast is enabled while the BSDF matrix branch is disabled; "
            "verify that the coarse non-diffuse BSDF contribution is removed elsewhere."
        )
    if has_bsdf and not any(group.mode == "bsdf" for group in groups):
        warnings.append(
            "A BSDF material was detected, but no [windowgroup:*] uses mode=bsdf. "
            "The scheduler cannot construct matched TDS subtraction matrices for it."
        )
    if ev_strategy == "bsdf_sun" and not bsdf_matrix:
        warnings.append("bsdf_sun was selected without an enabled BSDF matrix branch.")

    return WorkflowDecision(
        representation=representation,
        ev_strategy=ev_strategy,
        directlobecontrast=direct,
        specularcontrast=specular,
        bsdf_matrix=bsdf_matrix,
        ttsuncontrast=ttsun,
        reasons={
            "directlobecontrast": direct_reason,
            "specularcontrast": specular_reason,
            "bsdf_matrix": bsdf_matrix_reason,
            "ttsuncontrast": ttsun_reason,
            "ev_strategy": ev_reason,
        },
        warnings=warnings,
    )


def count_views(path_value: str) -> int:
    path = host_path(path_value)
    try:
        count = sum(
            1
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except OSError as exc:
        raise SchedulerError("Could not count viewpoints in {}: {}".format(path, exc))
    if count <= 0:
        raise SchedulerError("No viewpoints were found in {}".format(path))
    return count


def required_path(settings: Settings, section: str, option: str) -> str:
    value = settings.path_value(section, option)
    if not value:
        raise SchedulerError("{}.{} is required".format(section, option))
    return value


class PlanBuilder:
    def __init__(
        self,
        settings: Settings,
        detection: DetectionResult,
        decision: WorkflowDecision,
        groups: List[WindowGroup],
    ) -> None:
        self.s = settings
        self.detection = detection
        self.decision = decision
        self.groups = groups
        self.steps: List[Step] = []
        self.warnings: List[str] = list(detection.warnings) + list(decision.warnings)
        self.out = required_path(settings, "scene", "out")
        self.views = required_path(settings, "scene", "views")
        configured_nview = settings.get("scene", "nview", "auto").lower()
        if configured_nview == "auto":
            self.nview = count_views(self.views)
        else:
            try:
                self.nview = int(configured_nview)
            except ValueError as exc:
                raise SchedulerError("scene.nview must be auto or an integer") from exc
        self.nproc = settings.getint("scene", "nproc", os.cpu_count() or 1)
        self.weather = required_path(settings, "scene", "weather")
        self.sky_receiver = required_path(settings, "scene", "sky_receiver")
        self.materials = settings.path_list("scene", "materials")
        self.black_materials = settings.path_list("scene", "black_materials")
        self.scene = settings.path_list("scene", "scene")
        if not self.materials or not self.black_materials or not self.scene:
            raise SchedulerError(
                "scene.materials, scene.black_materials, and scene.scene must each contain at least one path"
            )
        self.mf = settings.getint("radiance", "mf", 1)
        self.nsky = 144 * self.mf * self.mf + 1
        self.photometric = settings.get("radiance", "photometric_weights", "47.4 119.9 11.6")
        if len(split_config_list(self.photometric)) != 3:
            raise SchedulerError("radiance.photometric_weights must contain three values")
        self._group_results: Dict[str, Dict[str, str]] = {}
        self._bsdf_transfer_cache: Dict[Tuple[object, ...], Tuple[str, str]] = {}
        self.include_direct_ev, _ = auto_bool(
            settings,
            "directlobecontrast",
            "include_illuminance",
            decision.ev_strategy == "path_components",
        )
        self.include_specular_ev, _ = auto_bool(
            settings,
            "specularcontrast",
            "include_illuminance",
            decision.ev_strategy == "path_components" and not decision.bsdf_matrix,
        )
        if decision.ev_strategy == "total":
            self.include_direct_ev = False
            self.include_specular_ev = False
        self.replace_ev_direct = (
            decision.ev_strategy == "bsdf_sun"
            or (decision.directlobecontrast and self.include_direct_ev)
            or (decision.specularcontrast and self.include_specular_ev)
        )

    def p(self, name: str) -> str:
        return self.out.rstrip("/\\") + "/" + name

    def add(
        self,
        step_id: str,
        stage: str,
        description: str,
        command: str,
        outputs: Sequence[str],
        inputs: Sequence[str] = (),
        tools: Sequence[str] = (),
    ) -> None:
        self.steps.append(
            Step(
                step_id=clean_id(step_id),
                stage=stage,
                description=description,
                command=command.strip(),
                outputs=list(outputs),
                inputs=list(inputs),
                tools=list(tools),
            )
        )

    def command_with_extra(self, section: str, base: str) -> str:
        extra = self.s.get(section, "extra_args", "")
        return base + ((" " + extra) if extra else "")

    def build(self) -> Plan:
        self.add_weather()
        self.add_octrees()
        self.add_background()
        for group in self.groups:
            self.add_window_group(group)
        contrast_files: List[str] = []
        ev_additions: List[str] = []
        if self.decision.directlobecontrast:
            contrast, illuminance = self.add_directlobe()
            contrast_files.append(contrast)
            if self.include_direct_ev:
                ev_additions.append(illuminance)
        if self.decision.ttsuncontrast:
            contrast_files.append(self.add_ttsuncontrast())
        if self.decision.specularcontrast:
            contrast, illuminance = self.add_specularcontrast()
            contrast_files.append(contrast)
            if illuminance:
                ev_additions.append(illuminance)
        if self.decision.ev_strategy == "bsdf_sun":
            ev_additions.append(self.add_bsdf_sun_illuminance())
        ev = self.add_ev_assembly(ev_additions)
        contrast = self.add_contrast_assembly(contrast_files)
        self.add_dcglare(ev, contrast)
        return Plan(
            config_path=str(self.s.path),
            config_hash=self.s.hash(),
            output_dir=self.out,
            execution_dir=self.s.path_value("scene", "root", str(self.s.base_dir)),
            path_prepend=self.s.get("runtime", "path_prepend", ""),
            nview=self.nview,
            quality=self.s.quality,
            detection=self.detection,
            decision=self.decision,
            groups=self.groups,
            steps=self.steps,
            warnings=self.warnings,
        )

    def add_weather(self) -> None:
        gendaymtx = self.s.command("gendaymtx")
        sky = self.p("sky_mf{}.smx".format(self.mf))
        direct = self.p("sky_direct_mf{}.smx".format(self.mf))
        suns = self.p("suns.rad")
        sunmods = self.p("suns.mod")
        sunsglow = self.p("sunsglow.rad")
        index = self.p("index.mod")
        command = """{cmd} -u -of -D {suns} -M {mods} -G {glow} -I {index} -m {mf} {wea} > {sky}
{cmd} -u -m {mf} -d {wea} > {direct}""".format(
            cmd=q(gendaymtx),
            suns=q(suns),
            mods=q(sunmods),
            glow=q(sunsglow),
            index=q(index),
            mf=self.mf,
            wea=q(self.weather),
            sky=q(sky),
            direct=q(direct),
        )
        self.add(
            "weather",
            "prepare",
            "Generate annual all-sky and direct-sun matrices",
            command,
            [sky, direct, suns, sunmods],
            [self.weather],
            [gendaymtx],
        )

    def add_octrees(self) -> None:
        oconv = self.s.command("oconv")
        model = self.p("model.oct")
        black = self.p("model_black.oct")
        command = "{oconv} -f {inputs} > {output}".format(
            oconv=q(oconv), inputs=join_args(self.materials + self.scene), output=q(model)
        )
        self.add(
            "octree_model",
            "prepare",
            "Compile the full scene octree",
            command,
            [model],
            self.materials + self.scene,
            [oconv],
        )
        command = "{oconv} -f {inputs} > {output}".format(
            oconv=q(oconv),
            inputs=join_args(self.black_materials + self.scene),
            output=q(black),
        )
        self.add(
            "octree_black",
            "prepare",
            "Compile the black-interior scene octree",
            command,
            [black],
            self.black_materials + self.scene,
            [oconv],
        )
        if (
            self.decision.directlobecontrast
            or self.decision.specularcontrast
            or self.decision.ttsuncontrast
        ):
            sun_octree = self.p("sun_explicit.oct")
            command = "{oconv} -f {inputs} > {output}".format(
                oconv=q(oconv),
                inputs=join_args(self.materials + [self.p("suns.rad")] + self.scene),
                output=q(sun_octree),
            )
            self.add(
                "octree_sun",
                "prepare",
                "Compile the scene with explicit annual solar sources",
                command,
                [sun_octree],
                self.materials + self.scene + [self.p("suns.rad")],
                [oconv],
            )

    def add_background(self) -> None:
        rfluxmtx = self.s.command("rfluxmtx")
        dctimestep = self.s.command("dctimestep")
        rmtxop = self.s.command("rmtxop")
        ab = self.s.getint("radiance", "background_ab", 5)
        direct_ab = self.s.getint("radiance", "background_direct_ab", 1)
        ad = self.s.getint("radiance", "background_ad", 50000)
        lw = self.s.getfloat("radiance", "background_lw", 0.00002)
        total_coeff = self.p("Ev_total_coeff.mtx")
        direct_coeff = self.p("Ev_direct_coeff.mtx")
        total = self.p("Ev_total.mtx")
        direct = self.p("Ev_direct_coarse.mtx")
        background = self.p("Ev_background.mtx")
        sky = self.p("sky_mf{}.smx".format(self.mf))
        direct_sky = self.p("sky_direct_mf{}.smx".format(self.mf))
        weights = " ".join(self.photometric.split())

        if self.decision.ev_strategy == "path_components":
            dual = self.p("Ev_dual_raw.mtx")
            ground_total = self.p("Ev_ground_total.dat")
            ground_direct = self.p("Ev_ground_direct.dat")
            sky_total = self.p("Ev_sky_total.dat")
            sky_direct = self.p("Ev_sky_direct.dat")
            command = """{rflux} -u- -I+ -y {nview} -ab {ab} -ad {ad} -lw {lw} -n {nproc} \\
  --path-components total,direct --path-direct-max-ambient 0 \\
  --path-direct-max-reflections 2 - {receiver} -i {model} < {views} > {dual}
{rcrop} 0 0 {nview} 1 {dual} | {getinfo} - > {gt}
{rcrop} 0 1 {nview} 1 {dual} | {getinfo} - > {gd}
{rcrop} 0 2 {nview} {nsky} {dual} | {getinfo} - > {st}
{rcrop} 0 {direct_start} {nview} {nsky} {dual} | {getinfo} - > {sd}
{rlam} {gt} {st} | {rcollate} -hi -fa3 -or {nview} -oc {ncols} > {tc}
{rlam} {gd} {sd} | {rcollate} -hi -fa3 -or {nview} -oc {ncols} > {dc}""".format(
                rflux=q(rfluxmtx),
                nview=self.nview,
                ab=ab,
                ad=ad,
                lw=lw,
                nproc=self.nproc,
                receiver=q(self.sky_receiver),
                model=q(self.p("model.oct")),
                views=q(self.views),
                dual=q(dual),
                rcrop=q(self.s.command("rcrop")),
                getinfo=q(self.s.command("getinfo")),
                rlam=q(self.s.command("rlam")),
                rcollate=q(self.s.command("rcollate")),
                gt=q(ground_total),
                gd=q(ground_direct),
                st=q(sky_total),
                sd=q(sky_direct),
                nsky=self.nsky,
                direct_start=2 + self.nsky,
                ncols=1 + self.nsky,
                tc=q(total_coeff),
                dc=q(direct_coeff),
            )
            self.add(
                "ev_coefficients",
                "background",
                "Calculate path-separated vertical-eye-illuminance coefficients",
                command,
                [total_coeff, direct_coeff],
                [self.p("model.oct"), self.views, self.sky_receiver],
                [rfluxmtx, "rcrop", "getinfo", "rlam", "rcollate"],
            )
        else:
            command = """{rflux} -u- -I+ -y {nview} -ab {ab} -ad {ad} -lw {lw} -n {nproc} \\
  - {receiver} -i {model} < {views} > {tc}
{rflux} -u- -I+ -y {nview} -ab {direct_ab} -ad {ad} -lw {lw} -n {nproc} \\
  - {receiver} -i {black} < {views} > {dc}""".format(
                rflux=q(rfluxmtx),
                nview=self.nview,
                ab=ab,
                direct_ab=direct_ab,
                ad=ad,
                lw=lw,
                nproc=self.nproc,
                receiver=q(self.sky_receiver),
                model=q(self.p("model.oct")),
                black=q(self.p("model_black.oct")),
                views=q(self.views),
                tc=q(total_coeff),
                dc=q(direct_coeff),
            )
            self.add(
                "ev_coefficients",
                "background",
                "Calculate total and coarse-direct vertical-eye-illuminance coefficients",
                command,
                [total_coeff, direct_coeff],
                [self.p("model.oct"), self.p("model_black.oct"), self.views],
                [rfluxmtx],
            )

        if self.replace_ev_direct:
            background_command = "{rmtx} -fa {total} + -s -1 {direct} > {background}".format(
                rmtx=q(rmtxop), total=q(total), direct=q(direct), background=q(background)
            )
            background_description = (
                "Evaluate annual background illuminance and remove the matched coarse direct component"
            )
        else:
            background_command = "{rmtx} -fa {total} > {background}".format(
                rmtx=q(rmtxop), total=q(total), background=q(background)
            )
            background_description = (
                "Evaluate annual background illuminance without explicit direct replacement"
            )
        command = """{dct} {tc} {sky} | {rmtx} -fa -c {weights} - > {total}
{dct} {dc} {direct_sky} | {rmtx} -fa -c {weights} - > {direct}
{background_command}""".format(
            dct=q(dctimestep),
            tc=q(total_coeff),
            dc=q(direct_coeff),
            sky=q(sky),
            direct_sky=q(direct_sky),
            rmtx=q(rmtxop),
            weights=weights,
            total=q(total),
            direct=q(direct),
            background_command=background_command,
        )
        self.add(
            "ev_background",
            "background",
            background_description,
            command,
            [total, direct, background],
            [total_coeff, direct_coeff, sky, direct_sky],
            [dctimestep, rmtxop],
        )

    def compile_group_open_octrees(self, group: WindowGroup) -> Tuple[str, str]:
        real_materials = group.open_materials or self.materials
        black_materials = group.open_black_materials or self.black_materials
        open_scene = group.open_scene
        if not open_scene:
            raise SchedulerError(
                "windowgroup:{} uses mode=bsdf and requires open_scene, unless precomputed "
                "tds_total and tds_direct matrices are supplied".format(group.name)
            )
        oconv = self.s.command("oconv")
        prefix = clean_id(group.name)
        real_octree = self.p("{}_open.oct".format(prefix))
        black_octree = self.p("{}_open_black.oct".format(prefix))
        command = """{oconv} -f {real_inputs} > {real_octree}
{oconv} -f {black_inputs} > {black_octree}""".format(
            oconv=q(oconv),
            real_inputs=join_args(real_materials + open_scene),
            real_octree=q(real_octree),
            black_inputs=join_args(black_materials + open_scene),
            black_octree=q(black_octree),
        )
        self.add(
            "{}_open_octrees".format(prefix),
            "window",
            "Compile open-aperture octrees for BSDF window group {}".format(group.name),
            command,
            [real_octree, black_octree],
            real_materials + black_materials + open_scene,
            [oconv],
        )
        return real_octree, black_octree

    def add_window_group(self, group: WindowGroup) -> None:
        prefix = clean_id(group.name)
        if group.tds_total and group.tds_direct:
            tds_total = group.tds_total
            tds_direct = group.tds_direct
            real_octree = self.p("model.oct")
            black_octree = self.p("model_black.oct")
        elif group.mode == "precomputed":
            raise SchedulerError(
                "windowgroup:{} uses mode=precomputed but tds_total or tds_direct is missing".format(
                    group.name
                )
            )
        elif group.mode == "bsdf":
            if not group.xml:
                raise SchedulerError("windowgroup:{} requires an XML BSDF".format(group.name))
            real_octree, black_octree = self.compile_group_open_octrees(group)
            tds_total, tds_direct = self.add_bsdf_group_daylight(
                group, real_octree, black_octree
            )
        else:
            real_octree = self.p("model.oct")
            black_octree = self.p("model_black.oct")
            tds_total, tds_direct = self.add_geometry_group_daylight(group)

        rfluxmtx = self.s.command("rfluxmtx")
        ab = self.s.getint("radiance", "view_ab", 1)
        ad = self.s.getint("radiance", "view_ad", 40000)
        lw = self.s.getfloat("radiance", "view_lw", 0.000025)
        vtotal = self.p("Vtotal_{}.mtx".format(prefix))
        vdirect = self.p("Vdirect_{}.mtx".format(prefix))
        command = """{rflux} -v -u- -y {nview} -I+ -ab {ab} -ad {ad} -lw {lw} -n {nproc} \\
  -faf - {receiver} -i {real_octree} < {views} > {vtotal}
{rflux} -v -u- -y {nview} -I+ -ab {ab} -ad {ad} -lw {lw} -n {nproc} \\
  -faf - {receiver} -i {black_octree} < {views} > {vdirect}""".format(
            rflux=q(rfluxmtx),
            nview=self.nview,
            ab=ab,
            ad=ad,
            lw=lw,
            nproc=self.nproc,
            receiver=q(group.receiver),
            real_octree=q(real_octree),
            black_octree=q(black_octree),
            views=q(self.views),
            vtotal=q(vtotal),
            vdirect=q(vdirect),
        )
        self.add(
            "view_{}".format(prefix),
            "window",
            "Calculate view matrices for window group {}".format(group.name),
            command,
            [vtotal, vdirect],
            [real_octree, black_octree, group.receiver, self.views],
            [rfluxmtx],
        )
        self._group_results[group.name] = {
            "receiver": group.receiver,
            "vtotal": vtotal,
            "vdirect": vdirect,
            "tds_total": tds_total,
            "tds_direct": tds_direct,
        }

    def add_geometry_group_daylight(self, group: WindowGroup) -> Tuple[str, str]:
        prefix = clean_id(group.name)
        rfluxmtx = self.s.command("rfluxmtx")
        dctimestep = self.s.command("dctimestep")
        ab = self.s.getint("radiance", "daylight_ab", 5)
        ad = self.s.getint("radiance", "daylight_ad", 1000)
        lw = self.s.getfloat("radiance", "daylight_lw", 0.001)
        rays = self.s.getint("radiance", "daylight_c", 250)
        direct_ss = self.s.getint("radiance", "daylight_direct_ss", 16)
        total_d = self.p("Dtotal_{}.dmx".format(prefix))
        direct_d = self.p("Ddirect_{}.dmx".format(prefix))
        total = self.p("TDStotal_{}.mtx".format(prefix))
        direct = self.p("TDSdirect_{}.mtx".format(prefix))
        command = """{rflux} -v -u- -ff -ab {ab} -ad {ad} -lw {lw} -c {rays} -n {nproc} \\
  {receiver} {sky_receiver} -i {model} > {total_d}
{rflux} -v -u- -ff -ab 0 -ad {ad} -ss {ss} -st 0 -lr 2 -lw {lw} \\
  -c {rays} -n {nproc} --direct-specular-only {receiver} {sky_receiver} \\
  -i {model} > {direct_d}""".format(
            rflux=q(rfluxmtx),
            ab=ab,
            ad=ad,
            lw=lw,
            rays=rays,
            nproc=self.nproc,
            receiver=q(group.receiver),
            sky_receiver=q(self.sky_receiver),
            model=q(self.p("model.oct")),
            total_d=q(total_d),
            direct_d=q(direct_d),
            ss=direct_ss,
        )
        self.add(
            "daylight_{}".format(prefix),
            "window",
            "Calculate total and non-diffuse daylight matrices for {}".format(group.name),
            command,
            [total_d, direct_d],
            [group.receiver, self.sky_receiver, self.p("model.oct")],
            [rfluxmtx],
        )
        command = """{dct} -of {total_d} {sky} > {total}
{dct} -of {direct_d} {direct_sky} > {direct}""".format(
            dct=q(dctimestep),
            total_d=q(total_d),
            direct_d=q(direct_d),
            sky=q(self.p("sky_mf{}.smx".format(self.mf))),
            direct_sky=q(self.p("sky_direct_mf{}.smx".format(self.mf))),
            total=q(total),
            direct=q(direct),
        )
        self.add(
            "annual_window_{}".format(prefix),
            "window",
            "Evaluate annual window-group luminance matrices for {}".format(group.name),
            command,
            [total, direct],
            [total_d, direct_d],
            [dctimestep],
        )
        return total, direct

    def add_bsdf_group_daylight(
        self, group: WindowGroup, real_octree: str, black_octree: str
    ) -> Tuple[str, str]:
        prefix = clean_id(group.name)
        rfluxmtx = self.s.command("rfluxmtx")
        rmtxop = self.s.command("rmtxop")
        dctimestep = self.s.command("dctimestep")
        ab = self.s.getint("radiance", "daylight_ab", 5)
        ad = self.s.getint("radiance", "daylight_ad", 1000)
        lw = self.s.getfloat("radiance", "daylight_lw", 0.001)
        rays = self.s.getint("radiance", "daylight_c", 250)
        total_d = self.p("Dopen_total_{}.dmx".format(prefix))
        direct_d = self.p("Dopen_direct_{}.dmx".format(prefix))
        tall, tdiff = self.bsdf_transfer_matrices(group)
        tdtotal = self.p("TDtotal_{}.dmx".format(prefix))
        tddirect = self.p("TDdirect_all_{}.dmx".format(prefix))
        tddiffuse = self.p("TDdiffuse_{}.dmx".format(prefix))
        total = self.p("TDStotal_{}.mtx".format(prefix))
        direct_all = self.p("TDSdirect_all_{}.mtx".format(prefix))
        diffuse = self.p("TDSdiffuse_{}.mtx".format(prefix))
        nondiffuse = self.p("TDSdirect_nondiffuse_{}.mtx".format(prefix))
        command = """{rflux} -v -u- -ff -ab {ab} -ad {ad} -lw {lw} -c {rays} -n {nproc} \\
  {receiver} {sky_receiver} -i {real_octree} > {total_d}
{rflux} -v -u- -ff -ab 0 -ad {ad} -lw {lw} -c {rays} -n {nproc} \\
  {receiver} {sky_receiver} -i {black_octree} > {direct_d}""".format(
            rflux=q(rfluxmtx),
            ab=ab,
            ad=ad,
            lw=lw,
            rays=rays,
            nproc=self.nproc,
            receiver=q(group.receiver),
            sky_receiver=q(self.sky_receiver),
            real_octree=q(real_octree),
            black_octree=q(black_octree),
            total_d=q(total_d),
            direct_d=q(direct_d),
        )
        self.add(
            "open_daylight_{}".format(prefix),
            "bsdf",
            "Calculate open-aperture daylight matrices for {}".format(group.name),
            command,
            [total_d, direct_d],
            [group.receiver, self.sky_receiver, real_octree, black_octree],
            [rfluxmtx],
        )
        command = """{rmtx} -ff {tall} . {total_d} > {tdtotal}
{dct} -of {tdtotal} {sky} > {total}
{rmtx} -ff {tall} . {direct_d} > {tddirect}
{rmtx} -ff {tdiff} . {direct_d} > {tddiffuse}
{dct} -of {tddirect} {direct_sky} > {direct_all}
{dct} -of {tddiffuse} {direct_sky} > {diffuse}
{rmtx} -ff {direct_all} + -s -1 {diffuse} > {nondiffuse}""".format(
            rmtx=q(rmtxop),
            dct=q(dctimestep),
            tall=q(tall),
            tdiff=q(tdiff),
            total_d=q(total_d),
            direct_d=q(direct_d),
            tdtotal=q(tdtotal),
            tddirect=q(tddirect),
            tddiffuse=q(tddiffuse),
            sky=q(self.p("sky_mf{}.smx".format(self.mf))),
            direct_sky=q(self.p("sky_direct_mf{}.smx".format(self.mf))),
            total=q(total),
            direct_all=q(direct_all),
            diffuse=q(diffuse),
            nondiffuse=q(nondiffuse),
        )
        self.add(
            "annual_bsdf_{}".format(prefix),
            "bsdf",
            "Assemble annual BSDF luminance and matched non-diffuse subtraction for {}".format(
                group.name
            ),
            command,
            [total, nondiffuse],
            [tall, tdiff, total_d, direct_d],
            [rmtxop, dctimestep],
        )
        return total, nondiffuse

    def bsdf_transfer_matrices(self, group: WindowGroup) -> Tuple[str, str]:
        samples = self.s.getint("bsdf", "bsdf_matrix_samples", 1024)
        diffuse_samples = self.s.getint("bsdf", "diffuse_matrix_samples", 1)
        seed = self.s.getint("bsdf", "matrix_seed", 19)
        direction = self.s.get("bsdf", "direction", "tb")
        key: Tuple[object, ...] = (
            group.xml,
            self.mf,
            samples,
            diffuse_samples,
            seed,
            direction,
        )
        cached = self._bsdf_transfer_cache.get(key)
        if cached:
            return cached
        digest = hashlib.sha1("|".join(str(item) for item in key).encode("utf-8")).hexdigest()[:10]
        tall = self.p("Tall_{}_mf{}.mtx".format(digest, self.mf))
        tdiff = self.p("Tdiffuse_{}_mf{}.mtx".format(digest, self.mf))
        bsdf2reinhart = self.s.command("bsdf2reinhart")
        command = """{tool} -q -ff -{direction} -mf {mf} -n {samples} -s {seed} -C all {xml} {tall}
{tool} -q -ff -{direction} -mf {mf} -n {diffuse_samples} -s {seed} -C diffuse {xml} {tdiff}""".format(
            tool=q(bsdf2reinhart),
            direction=direction,
            mf=self.mf,
            samples=samples,
            diffuse_samples=diffuse_samples,
            seed=seed,
            xml=q(group.xml),
            tall=q(tall),
            tdiff=q(tdiff),
        )
        self.add(
            "bsdf_resample_{}".format(digest),
            "bsdf",
            "Resample and cache complete and diffuse BSDF components",
            command,
            [tall, tdiff],
            [group.xml],
            [bsdf2reinhart],
        )
        self._bsdf_transfer_cache[key] = (tall, tdiff)
        return tall, tdiff

    def add_directlobe(self) -> Tuple[str, str]:
        tool = self.s.command("directlobecontrast")
        output = self.p("directlobe_contrast.mtx")
        sun = self.p("direct_sun_contrast.mtx")
        lobe = self.p("direct_lobe_contrast.mtx")
        irradiance = self.p("direct_sun_irradiance.mtx")
        illuminance = self.p("direct_sun_illuminance.mtx")
        sun_samples = self.s.getint("directlobecontrast", "sun_samples", 10000)
        lobe_samples = self.s.getint("directlobecontrast", "lobe_samples", 16)
        disk_res = self.s.getint("directlobecontrast", "lobe_disk_resolution", 8)
        pilot_res = self.s.getint(
            "directlobecontrast", "lobe_disk_pilot_resolution", 0
        )
        guard = self.s.getfloat(
            "directlobecontrast", "lobe_disk_pilot_guard_angle", 1.0
        )
        base = """{tool} -q -vf {views} -S {suns} \\
  --sun-output {sun} --lobe-output {lobe} \\
  --sun-irradiance-output {irradiance} --sun-mode batch \\
  --sun-samples {sun_samples} --lobe-samples {lobe_samples} \\
  --lobe-sun-disk-resolution {disk_res} \\
  --lobe-sun-disk-pilot-resolution {pilot_res} \\
  --lobe-sun-disk-pilot-guard-angle {guard} \\
  -n {nproc} -o {output} {octree}""".format(
            tool=q(tool),
            views=q(self.views),
            suns=q(self.p("suns.rad")),
            sun=q(sun),
            lobe=q(lobe),
            irradiance=q(irradiance),
            sun_samples=sun_samples,
            lobe_samples=lobe_samples,
            disk_res=disk_res,
            pilot_res=pilot_res,
            guard=guard,
            nproc=self.nproc,
            output=q(output),
            octree=q(self.p("sun_explicit.oct")),
        )
        base = self.command_with_extra("directlobecontrast", base)
        self.add(
            "directlobecontrast",
            "contrast",
            "Resolve the finite solar disk and analytical transmission/refraction lobes",
            base,
            [output, sun, lobe, irradiance],
            [self.views, self.p("suns.rad"), self.p("sun_explicit.oct")],
            [tool],
        )
        rmtxop = self.s.command("rmtxop")
        weights = " ".join(self.photometric.split())
        command = "{rmtx} -fa -c {weights} {source} > {output}".format(
            rmtx=q(rmtxop), weights=weights, source=q(irradiance), output=q(illuminance)
        )
        self.add(
            "direct_sun_illuminance",
            "assemble",
            "Convert direct-sun irradiance to vertical eye illuminance",
            command,
            [illuminance],
            [irradiance],
            [rmtxop],
        )
        return output, illuminance

    def add_ttsuncontrast(self) -> str:
        tool = self.s.command("ttsuncontrast")
        output = self.p("ttsun_contrast.mtx")
        samples = self.s.getint("ttsuncontrast", "bsdf_samples", 256)
        disk_samples = self.s.getint("ttsuncontrast", "sun_disk_samples", 1)
        seed = self.s.getint("ttsuncontrast", "sample_seed", 1)
        base = """{tool} -q -vf {views} -S {suns} --auto-bsdf {octree} \\
  --bsdf-samples {samples} --sun-disk-samples {disk_samples} \\
  --sample-seed {seed} -n {nproc} -o {output}""".format(
            tool=q(tool),
            views=q(self.views),
            suns=q(self.p("suns.rad")),
            octree=q(self.p("sun_explicit.oct")),
            samples=samples,
            disk_samples=disk_samples,
            seed=seed,
            nproc=self.nproc,
            output=q(output),
        )
        base = self.command_with_extra("ttsuncontrast", base)
        self.add(
            "ttsuncontrast",
            "contrast",
            "Resolve compact non-diffuse BSDF/aBSDF transmission peaks",
            base,
            [output],
            [self.views, self.p("suns.rad"), self.p("sun_explicit.oct")],
            [tool],
        )
        return output

    def add_specularcontrast(self) -> Tuple[str, str]:
        tool = self.s.command("specularcontrast")
        output = self.p("specular_contrast.mtx")
        illuminance = self.p("mirror_illuminance.mtx")
        normals = self.p("specular_normals.txt")
        paths = self.p("specular_paths.txt")
        disk = self.s.getint("specularcontrast", "sun_disk_samples", 64)
        secondary_disk = self.s.getint(
            "specularcontrast", "secondary_sun_disk_samples", 16
        )
        rough = self.s.getint("specularcontrast", "rough_samples", 1024)
        rough_secondary = self.s.getint(
            "specularcontrast", "rough_secondary_samples", 16
        )
        level = self.s.getint("specularcontrast", "reflection_level", 5)
        max_bounces = self.s.getint("specularcontrast", "max_specular_bounces", 2)
        block = self.s.getint("specularcontrast", "block_size", 2048)
        threshold = self.s.getfloat("specularcontrast", "threshold", 2000.0)
        allowlist = self.s.path_value("specularcontrast", "allowlist")
        auto_args = "--auto-materials"
        if allowlist:
            auto_args += " --auto-material-allowlist {}".format(q(allowlist))
        elif self.detection.specular_materials:
            auto_args += " --auto-material-allowlist {}".format(
                q(self.p("detected_specular.mod"))
            )
        mirror_output = ""
        if self.include_specular_ev:
            mirror_output = "--mirror-illuminance-output {}".format(q(illuminance))
        base = """{tool} -q -vf {views} -S {suns} {auto_args} \\
  --sun-disk-samples {disk} --secondary-sun-disk-samples {secondary_disk} \\
  --sun-disk-seed 0 --rough-samples {rough} \\
  --rough-secondary-samples {rough_secondary} --rough-seed 0 \\
  --reflection-level {level} --reflection-seed 0 --save-normals {normals} \\
  --max-specular-bounces {max_bounces} --save-reflection-paths {paths} \\
  --direct-specular-only --adaptive-sun-disk --sun-disk-pilot-samples 16 \\
  --secondary-sun-disk-pilot-samples 4 --sun-disk-pilot-guard-angle 8 \\
  --adaptive-rough-sampling --rough-pilot-samples 64 \\
  --rough-pilot-threshold 1e-6 --rough-pilot-guard-angle 5 \\
  --rough-pilot-anchor-angle 5 --adaptive-rough-cells \\
  --rough-medium-samples 64 --rough-cell-trigger 0.5 \\
  --normal-tolerance 0.25 {mirror_output} -n {nproc} -b {block} -t {threshold} \\
  {octree} > {output}""".format(
            tool=q(tool),
            views=q(self.views),
            suns=q(self.p("suns.rad")),
            auto_args=auto_args,
            disk=disk,
            secondary_disk=secondary_disk,
            rough=rough,
            rough_secondary=rough_secondary,
            level=level,
            normals=q(normals),
            max_bounces=max_bounces,
            paths=q(paths),
            mirror_output=mirror_output,
            nproc=self.nproc,
            block=block,
            threshold=threshold,
            octree=q(self.p("sun_explicit.oct")),
            output=q(output),
        )
        base = self.command_with_extra("specularcontrast", base)
        outputs = [output]
        if mirror_output:
            outputs.append(illuminance)
        allowlist_input = (
            allowlist
            if allowlist
            else self.p("detected_specular.mod")
            if self.detection.specular_materials
            else ""
        )
        self.add(
            "specularcontrast",
            "contrast",
            "Resolve selected first- and second-order specular solar images",
            base,
            outputs,
            [
                self.views,
                self.p("suns.rad"),
                self.p("sun_explicit.oct"),
            ]
            + ([allowlist_input] if allowlist_input else []),
            [tool],
        )
        return output, illuminance if mirror_output else ""

    def add_bsdf_sun_illuminance(self) -> str:
        rcontrib = self.s.command("rcontrib")
        rmtxop = self.s.command("rmtxop")
        oconv = self.s.command("oconv")
        samples = self.s.getint("bsdf", "sun_samples", 1000)
        ad = self.s.getint("bsdf", "sun_ad", 10000)
        lw = self.s.getfloat("bsdf", "sun_lw", 0.0001)
        black_sun_octree = self.p("sun_black.oct")
        accurate = self.p("Ev_bsdf_sun_accurate.mtx")
        coarse_direct = self.p("Ev_bsdf_sun_coarse_direct.mtx")
        coarse_full = self.p("Ev_bsdf_sun_coarse_full.mtx")
        result = self.p("Ev_bsdf_sun.mtx")
        weights = " ".join(self.photometric.split())
        command = "{oconv} -f {inputs} > {output}".format(
            oconv=q(oconv),
            inputs=join_args(self.black_materials + self.scene + [self.p("suns.rad")]),
            output=q(black_sun_octree),
        )
        self.add(
            "octree_bsdf_sun_black",
            "prepare",
            "Compile the black-interior scene used for finite-sun BSDF illuminance",
            command,
            [black_sun_octree],
            self.black_materials + self.scene + [self.p("suns.rad")],
            [oconv],
        )
        command = """awk -v samples={samples} '{{for (i=0; i<samples; i++) print}}' {views} | \\
  {rcontrib} -u- -I+ -V+ -ab 0 -y {nview} -n {nproc} -ad {ad} -lw {lw} \\
  -dt 0 -dc 1 -dj 1 -dp 0 -dr 1 -faa -c {samples} -M {mods} {octree} | \\
  {rmtx} -fa -c {weights} - > {accurate}
{rcontrib} -u- -I+ -V+ -ab 0 -y {nview} -n {nproc} -ad {ad} -lw {lw} \\
  -dt 0 -dc 1 -dj 0 -dr 0 -faa -M {mods} {octree} < {views} | \\
  {rmtx} -fa -c {weights} - > {coarse_direct}
{rcontrib} -u- -I+ -V+ -ab 1 -y {nview} -n {nproc} -ad {ad} -lw {lw} \\
  -dt 0 -dc 1 -dj 0 -dr 0 -faa -M {mods} {octree} < {views} | \\
  {rmtx} -fa -c {weights} - > {coarse_full}
{rmtx} -fa {coarse_full} + -s -1 {coarse_direct} + {accurate} > {result}""".format(
            samples=samples,
            views=q(self.views),
            rcontrib=q(rcontrib),
            nview=self.nview,
            nproc=self.nproc,
            ad=ad,
            lw=lw,
            mods=q(self.p("suns.mod")),
            octree=q(black_sun_octree),
            rmtx=q(rmtxop),
            weights=weights,
            accurate=q(accurate),
            coarse_direct=q(coarse_direct),
            coarse_full=q(coarse_full),
            result=q(result),
        )
        self.add(
            "bsdf_sun_illuminance",
            "contrast",
            "Replace coarse BSDF direct-sun illuminance with finite-sun sampling",
            command,
            [accurate, coarse_direct, coarse_full, result],
            [black_sun_octree, self.views, self.p("suns.mod")],
            [rcontrib, rmtxop, "awk"],
        )
        return result

    def add_ev_assembly(self, additions: Sequence[str]) -> str:
        rmtxop = self.s.command("rmtxop")
        output = self.p("Ev.mtx")
        components = [self.p("Ev_background.mtx")] + [item for item in additions if item]
        expression = " + ".join(q(item) for item in components)
        command = "{tool} -fa {expression} > {output}".format(
            tool=q(rmtxop), expression=expression, output=q(output)
        )
        self.add(
            "assemble_ev",
            "assemble",
            "Assemble background and explicitly resolved vertical eye illuminance",
            command,
            [output],
            components,
            [rmtxop],
        )
        return output

    def add_contrast_assembly(self, contrasts: Sequence[str]) -> str:
        if not contrasts:
            return ""
        rmtxop = self.s.command("rmtxop")
        output = self.p("explicit_contrast_combined.mtx")
        expression = " + ".join(q(item) for item in contrasts)
        command = "{tool} -fa {expression} > {output}".format(
            tool=q(rmtxop), expression=expression, output=q(output)
        )
        self.add(
            "assemble_contrast",
            "assemble",
            "Combine the enabled source-specific DGP contrast terms",
            command,
            [output],
            list(contrasts),
            [rmtxop],
        )
        return output

    def add_dcglare(self, ev: str, contrast: str) -> None:
        tool = self.s.command("dcglare2")
        output = self.s.path_value("dcglare2", "output", self.p("dgp_annual.mtx"))
        parts = [q(tool), "-h", "-vf", q(self.views)]
        threshold = self.s.get("dcglare2", "threshold", "")
        if threshold:
            parts.extend(["-t", q(threshold)])
        for group in self.groups:
            result = self._group_results[group.name]
            parts.extend(
                [
                    "-wgroup",
                    q(result["receiver"]),
                    "-wVTotal",
                    q(result["vtotal"]),
                    "-wTDSTotal",
                    q(result["tds_total"]),
                ]
            )
            if contrast:
                parts.extend(
                    [
                        "-wVDirect",
                        q(result["vdirect"]),
                        "-wTDSDirect",
                        q(result["tds_direct"]),
                    ]
                )
        if contrast:
            parts.extend(["-wSpecularContrast", q(contrast)])
        extra = self.s.get("dcglare2", "extra_args", "")
        if extra:
            parts.append(extra)
        parts.extend([q(ev), ">", q(output)])
        command = (" " + "\\" + "\n  ").join(parts)
        inputs = [self.views, ev] + ([contrast] if contrast else [])
        for result in self._group_results.values():
            inputs.extend([result["vtotal"], result["tds_total"]])
            if contrast:
                inputs.extend([result["vdirect"], result["tds_direct"]])
        self.add(
            "dcglare2",
            "evaluate",
            "Calculate annual DGP from the coarse field and explicit source terms",
            command,
            [output],
            inputs,
            [tool],
        )


def build_plan(settings: Settings) -> Plan:
    detection = inspect_scene(settings)
    groups = read_window_groups(settings, detection)
    decision = choose_workflow(settings, detection, groups)
    return PlanBuilder(settings, detection, decision, groups).build()


def print_inspection(
    detection: DetectionResult,
    decision: WorkflowDecision,
    groups: Sequence[WindowGroup] = (),
) -> None:
    print("Representation: {}".format(decision.representation))
    print("Active materials: {}".format(", ".join(detection.active_material_names) or "none"))
    print(
        "Analytical transmission: {}".format(
            ", ".join(detection.analytic_transmission_materials) or "none"
        )
    )
    print("Specular: {}".format(", ".join(detection.specular_materials) or "none"))
    print("BSDF/aBSDF: {}".format(", ".join(detection.bsdf_materials + detection.absdf_materials) or "none"))
    for item in detection.bsdf_files:
        print("  BSDF {}: {}".format(item.path, item.representation))
    print(
        "Window-normal reference: {} ({})".format(
            detection.normal_reference,
            detection.normal_reference_count,
        )
    )
    print("Selected stages:")
    print("  directlobecontrast: {}".format(decision.directlobecontrast))
    print("  specularcontrast:   {}".format(decision.specularcontrast))
    print("  BSDF matrix:        {}".format(decision.bsdf_matrix))
    print("  ttsuncontrast:      {}".format(decision.ttsuncontrast))
    print("  Ev strategy:        {}".format(decision.ev_strategy))
    if groups:
        print("Window groups:")
        for group in groups:
            origin = "auto" if group.auto_detected else "configured"
            detail = ", ".join(group.materials) if group.materials else group.receiver
            print(
                "  {}: mode={}, origin={}, surfaces={}, optics={}".format(
                    group.name,
                    group.mode,
                    origin,
                    len(group.surface_names) if group.surface_names else "n/a",
                    detail,
                )
            )
    warnings = detection.warnings + decision.warnings
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print("  - {}".format(warning))


def render_shell_script(plan: Plan) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "mkdir -p {}".format(q(plan.output_dir)),
        "cd {}".format(q(plan.execution_dir)),
        "",
        "# Generated by eigcm_scheduler.py {}".format(VERSION),
        "# Representation: {}".format(plan.decision.representation),
        "# Quality: {}".format(plan.quality),
        "",
    ]
    if plan.path_prepend:
        lines.insert(2, "export PATH={}:$PATH".format(q(plan.path_prepend)))
    generated_groups = [group for group in plan.groups if group.receiver_content]
    if generated_groups:
        lines.extend(["# Materialize automatically generated window receivers", ""])
        for group in generated_groups:
            normalized_path = group.receiver.replace("\\", "/")
            parent = normalized_path.rsplit("/", 1)[0]
            delimiter = "EIGCM_RECEIVER_{}".format(clean_id(group.name).upper())
            lines.extend(
                [
                    "mkdir -p {}".format(q(parent)),
                    "cat > {} <<'{}'".format(q(group.receiver), delimiter),
                    group.receiver_content.rstrip(),
                    delimiter,
                    "",
                ]
            )
    for index, step in enumerate(plan.steps, start=1):
        lines.extend(
            [
                "# {:02d} {}: {}".format(index, step.step_id, step.description),
                step.command,
                "",
            ]
        )
    return "\n".join(lines)


def print_plan(plan: Plan, show_commands: bool = True) -> None:
    print("EIGCM workflow plan")
    print("  representation: {}".format(plan.decision.representation))
    print("  quality:        {}".format(plan.quality))
    print("  viewpoints:     {}".format(plan.nview))
    print("  output:         {}".format(plan.output_dir))
    print("  window groups:  {}".format(len(plan.groups)))
    print("  steps:          {}".format(len(plan.steps)))
    for index, step in enumerate(plan.steps, start=1):
        print("\n[{:02d}] {} ({})".format(index, step.step_id, step.stage))
        print("     {}".format(step.description))
        if show_commands:
            for line in step.command.splitlines():
                print("     {}".format(line))
    if plan.warnings:
        print("\nWarnings:")
        for warning in plan.warnings:
            print("  - {}".format(warning))


def ensure_output_files(plan: Plan) -> None:
    out = host_path(plan.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_generated_assets(plan)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (out / "eigcm_plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "eigcm_plan.sh").write_text(render_shell_script(plan), encoding="utf-8")
    if plan.detection.specular_materials:
        (out / "detected_specular.mod").write_text(
            "\n".join(plan.detection.specular_materials) + "\n", encoding="utf-8"
        )


def write_generated_assets(plan: Plan) -> None:
    for group in plan.groups:
        if not group.receiver_content:
            continue
        destination = host_path(group.receiver)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(group.receiver_content, encoding="ascii")


def step_is_current(step: Step) -> bool:
    if not step.outputs:
        return False
    outputs = [host_path(path) for path in step.outputs]
    if not all(path.exists() and path.stat().st_size > 0 for path in outputs):
        return False
    inputs = [host_path(path) for path in step.inputs]
    existing_inputs = [path for path in inputs if path.exists()]
    if not existing_inputs:
        return True
    newest_input = max(path.stat().st_mtime for path in existing_inputs)
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return oldest_output >= newest_input


def shell_invocation(shell: str, command: str) -> List[str]:
    shell_name = Path(shell).name.lower()
    if "bash" in shell_name or "zsh" in shell_name or "sh" == shell_name:
        return [shell, "-o", "pipefail", "-c", command]
    return [shell, "-c", command]


def check_tools(settings: Settings, steps: Sequence[Step]) -> None:
    if settings.getbool("runtime", "skip_preflight", False):
        return
    tools = sorted({tool for step in steps for tool in step.tools})
    if not tools:
        return
    shell = settings.get("runtime", "shell", "bash")
    checks = "\n".join("command -v {} >/dev/null || echo {}".format(q(tool), q(tool)) for tool in tools)
    path_prepend = settings.get("runtime", "path_prepend", "")
    if path_prepend and "bash" in Path(shell).name.lower():
        checks = "export PATH={}:$PATH\n{}".format(q(path_prepend), checks)
    try:
        result = subprocess.run(
            shell_invocation(shell, checks),
            cwd=str(host_path(settings.path_value("scene", "root", str(settings.base_dir)))),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        raise SchedulerError(
            "Could not start the configured runtime shell {!r}: {}".format(shell, exc)
        ) from exc
    missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if missing:
        raise SchedulerError(
            "Required commands are not available in the configured shell: {}".format(
                ", ".join(missing)
            )
        )


def check_external_inputs(plan: Plan, steps: Sequence[Step]) -> None:
    missing: List[str] = []
    generated_before: Set[str] = set()
    for step in steps:
        for value in step.inputs:
            if value in generated_before:
                continue
            path = host_path(value)
            if not path.exists() and value not in missing:
                missing.append(value)
        generated_before.update(step.outputs)
    if missing:
        raise SchedulerError(
            "Required input files do not exist: {}".format(", ".join(missing))
        )


def selected_steps(
    steps: Sequence[Step], from_step: str = "", until_step: str = ""
) -> List[Step]:
    ids = [step.step_id for step in steps]
    start = 0
    end = len(steps)
    if from_step:
        if from_step not in ids:
            raise SchedulerError("Unknown --from step {!r}".format(from_step))
        start = ids.index(from_step)
    if until_step:
        if until_step not in ids:
            raise SchedulerError("Unknown --until step {!r}".format(until_step))
        end = ids.index(until_step) + 1
    if end < start:
        raise SchedulerError("--until occurs before --from")
    return list(steps[start:end])


def run_plan(
    settings: Settings,
    plan: Plan,
    force: bool = False,
    from_step: str = "",
    until_step: str = "",
) -> None:
    ensure_output_files(plan)
    shell = settings.get("runtime", "shell", "bash")
    resume = settings.getbool("runtime", "resume", True)
    env = os.environ.copy()
    path_prepend = settings.get("runtime", "path_prepend", "")
    bash_shell = "bash" in Path(shell).name.lower()
    if path_prepend and not bash_shell:
        env["PATH"] = path_prepend + os.pathsep + env.get("PATH", "")
    steps = selected_steps(plan.steps, from_step, until_step)
    check_external_inputs(plan, steps)
    check_tools(settings, steps)
    log_dir = host_path(plan.output_dir) / "logs"
    started = time.time()
    for index, step in enumerate(steps, start=1):
        if resume and not force and step_is_current(step):
            print("[{:02d}/{}] SKIP {} (outputs are current)".format(index, len(steps), step.step_id))
            continue
        print("[{:02d}/{}] RUN  {}: {}".format(index, len(steps), step.step_id, step.description))
        log_path = log_dir / "{}.log".format(step.step_id)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("# started {}\n".format(_dt.datetime.now().isoformat()))
            log.write("$ {}\n\n".format(step.command))
            try:
                runtime_command = step.command
                if path_prepend and bash_shell:
                    runtime_command = "export PATH={}:$PATH\n{}".format(
                        q(path_prepend), runtime_command
                    )
                process = subprocess.Popen(
                    shell_invocation(shell, runtime_command),
                    cwd=str(host_path(plan.execution_dir)),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                raise SchedulerError(
                    "Could not start step {} with shell {!r}: {}".format(
                        step.step_id, shell, exc
                    )
                ) from exc
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                log.write(line)
            return_code = process.wait()
            log.write("\n# return code {}\n".format(return_code))
        if return_code != 0:
            raise SchedulerError(
                "Step {} failed with exit code {}; see {}".format(
                    step.step_id, return_code, log_path
                )
            )
        missing = [
            path for path in step.outputs if not host_path(path).exists() or host_path(path).stat().st_size == 0
        ]
        if missing:
            raise SchedulerError(
                "Step {} completed but did not create: {}".format(step.step_id, ", ".join(missing))
            )
    elapsed = time.time() - started
    print("Completed {} selected steps in {:.1f} s".format(len(steps), elapsed))


def write_example(path: Path) -> None:
    if path.exists():
        raise SchedulerError("Refusing to overwrite existing file: {}".format(path))
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "eigcm_scheduler_example.cfg",
        script_dir.parent / "share" / "eigcm" / "eigcm_scheduler_example.cfg",
    ]
    template = next((candidate for candidate in candidates if candidate.exists()), None)
    if template is None:
        raise SchedulerError("Bundled example configuration is missing: {}".format(template))
    shutil.copyfile(str(template), str(path))
    print("Created {}".format(path))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically select and schedule an EIGCM annual glare workflow."
    )
    parser.add_argument("--version", action="version", version=VERSION)
    subparsers = parser.add_subparsers(dest="action", required=True)

    init_parser = subparsers.add_parser("init", help="copy the example configuration")
    init_parser.add_argument("output", help="new .cfg file")

    for name, help_text in (
        ("inspect", "inspect materials and explain the selected workflow"),
        ("plan", "build the workflow without running Radiance"),
        ("run", "execute the selected workflow"),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("config", help="EIGCM INI configuration")
        child.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="SECTION.OPTION=VALUE",
            help="override one configuration value; may be repeated",
        )
        if name in {"inspect", "plan"}:
            child.add_argument("--json", action="store_true", help="print JSON")
        if name == "plan":
            child.add_argument(
                "--summary", action="store_true", help="omit full command lines"
            )
            child.add_argument("--save", help="write a generated Bash script")
        if name == "run":
            child.add_argument("--force", action="store_true", help="rerun current steps")
            child.add_argument("--from", dest="from_step", default="", help="first step id")
            child.add_argument("--until", dest="until_step", default="", help="last step id")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            write_example(Path(args.output).resolve())
            return 0
        settings = Settings(args.config, args.overrides)
        detection = inspect_scene(settings)
        groups = read_window_groups(settings, detection)
        decision = choose_workflow(settings, detection, groups)
        if args.action == "inspect":
            if args.json:
                print(
                    json.dumps(
                        {
                            "detection": detection.to_dict(),
                            "decision": decision.to_dict(),
                            "window_groups": [dataclasses.asdict(group) for group in groups],
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print_inspection(detection, decision, groups)
            return 0
        plan = PlanBuilder(settings, detection, decision, groups).build()
        if args.action == "plan":
            if args.json:
                print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
            else:
                print_plan(plan, show_commands=not args.summary)
            if args.save:
                save_path = Path(args.save).resolve()
                write_generated_assets(plan)
                save_path.write_text(render_shell_script(plan), encoding="utf-8")
                print("Saved {}".format(save_path))
            return 0
        if args.action == "run":
            run_plan(
                settings,
                plan,
                force=args.force,
                from_step=args.from_step,
                until_step=args.until_step,
            )
            return 0
        parser.error("unknown action")
        return 2
    except SchedulerError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
