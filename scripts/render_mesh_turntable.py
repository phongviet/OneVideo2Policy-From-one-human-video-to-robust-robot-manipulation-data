"""Render a GLB turntable with a consistent camera and white background."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image


def camera_pose(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = position - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    pose[:3, 3] = position
    return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = trimesh.load(args.mesh, force="mesh")
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.7, 0.7, 0.7])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = pyrender.PerspectiveCamera(yfov=math.pi / 3)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    renderer = pyrender.OffscreenRenderer(256, 256)
    center = np.asarray(mesh.bounds).mean(axis=0)
    radius = max(mesh.extents) * 2.0
    sheet = Image.new("RGB", (4 * 256, 2 * 256), "white")
    for index in range(8):
        angle = index * math.tau / 8
        position = center + radius * np.array([math.sin(angle), 0.2, math.cos(angle)])
        pose = camera_pose(position, center)
        camera_node = scene.add(camera, pose=pose)
        light_node = scene.add(light, pose=pose)
        color, _ = renderer.render(scene)
        sheet.paste(Image.fromarray(color), ((index % 4) * 256, (index // 4) * 256))
        scene.remove_node(camera_node)
        scene.remove_node(light_node)
    renderer.delete()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
