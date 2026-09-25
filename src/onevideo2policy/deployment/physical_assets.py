"""Dependency-free printable geometry generation and STL auditing."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

BALL_DIAMETER_M = 0.03832122209971702
BOWL_OUTER_DIAMETER_M = 0.0991230365017926
BOWL_HEIGHT_M = 0.05693338151487156
BOWL_WALL_M = 0.004


def sphere_triangles(radius: float, segments: int, rings: int) -> np.ndarray:
    vertices = [[0.0, 0.0, -radius]]
    for ring in range(1, rings):
        latitude = -np.pi / 2 + np.pi * ring / rings
        radial = radius * np.cos(latitude)
        z = radius * np.sin(latitude)
        for segment in range(segments):
            angle = 2 * np.pi * segment / segments
            vertices.append([radial * np.cos(angle), radial * np.sin(angle), z])
    north = len(vertices)
    vertices.append([0.0, 0.0, radius])
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = []
    first = 1
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append([0, first + segment, first + nxt])
    for ring in range(rings - 2):
        current = 1 + ring * segments
        following = current + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.extend(
                (
                    [current + segment, following + segment, following + nxt],
                    [current + segment, following + nxt, current + nxt],
                )
            )
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append([last + segment, north, last + nxt])
    return vertices[np.asarray(faces)]


def bowl_triangles(outer_radius: float, height: float, wall: float, segments: int) -> np.ndarray:
    inner_radius = outer_radius - wall
    floor = wall
    vertices = []
    for radius, z in (
        (outer_radius, 0.0),
        (outer_radius, height),
        (inner_radius, floor),
        (inner_radius, height),
    ):
        for segment in range(segments):
            angle = 2 * np.pi * segment / segments
            vertices.append([radius * np.cos(angle), radius * np.sin(angle), z])
    bottom_center = len(vertices)
    vertices.append([0.0, 0.0, 0.0])
    floor_center = len(vertices)
    vertices.append([0.0, 0.0, floor])
    vertices = np.asarray(vertices, dtype=np.float64)
    outer_bottom, outer_top, inner_bottom, inner_top = (0, segments, 2 * segments, 3 * segments)
    faces = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        ob, obn = outer_bottom + segment, outer_bottom + nxt
        ot, otn = outer_top + segment, outer_top + nxt
        ib, ibn = inner_bottom + segment, inner_bottom + nxt
        it, itn = inner_top + segment, inner_top + nxt
        faces.extend(
            (
                [ob, obn, otn],
                [ob, otn, ot],
                [ib, itn, ibn],
                [ib, it, itn],
                [ot, otn, itn],
                [ot, itn, it],
                [bottom_center, obn, ob],
                [floor_center, ib, ibn],
            )
        )
    return vertices[np.asarray(faces)]


def write_binary_stl(path: Path, triangles_millimeters: np.ndarray) -> None:
    triangles = np.asarray(triangles_millimeters, dtype=np.float32)
    with path.open("wb") as stream:
        stream.write(b"OneVideo2Policy physical fixture".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = np.linalg.norm(normal)
            if norm:
                normal /= norm
            stream.write(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))


def audit_binary_stl(path: Path) -> dict:
    with path.open("rb") as stream:
        stream.read(80)
        count = struct.unpack("<I", stream.read(4))[0]
        triangles = np.empty((count, 3, 3), dtype=np.float32)
        for index in range(count):
            record = struct.unpack("<12fH", stream.read(50))
            triangles[index] = np.asarray(record[3:12]).reshape(3, 3)
        if stream.read(1):
            raise ValueError("unexpected trailing STL data")
    points = triangles.reshape(-1, 3)
    edges: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for triangle in triangles:
        keys = [tuple(float(value) for value in vertex) for vertex in triangle]
        for start, end in ((keys[0], keys[1]), (keys[1], keys[2]), (keys[2], keys[0])):
            edge = tuple(sorted((start, end)))
            edges[edge] = edges.get(edge, 0) + 1
    extents = points.max(axis=0) - points.min(axis=0)
    return {
        "triangles": count,
        "extents_mm": [float(value) for value in extents],
        "watertight": bool(edges and all(uses == 2 for uses in edges.values())),
        "boundary_or_nonmanifold_edges": sum(uses != 2 for uses in edges.values()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
