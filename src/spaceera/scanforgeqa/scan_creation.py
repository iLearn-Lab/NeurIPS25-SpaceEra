from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from spaceera.schemas import ScanSequence, ScanTrajectoryStep, SceneGraph, VideoFrame
from spaceera.utils import dump_json


def _build_orbit_steps(scene: SceneGraph, radius: float, orbit_views: int) -> List[ScanTrajectoryStep]:
    steps: List[ScanTrajectoryStep] = []
    objects = [obj for room in scene.rooms for obj in room.objects]
    if not objects:
        return steps

    anchor = objects[0]
    for index in range(orbit_views):
        yaw_deg = round((360.0 / orbit_views) * index, 1)
        position = [anchor.center[0], anchor.center[1], 1.6]
        steps.append(
            ScanTrajectoryStep(
                step_id=f"orbit_{index:03d}",
                action="orbit",
                position=position,
                yaw_deg=yaw_deg,
                note=f"Orbit around {anchor.object_id} with radius {radius:.2f} m.",
            )
        )
    return steps


def _build_navigation_steps(scene: SceneGraph, stride: float) -> List[ScanTrajectoryStep]:
    steps: List[ScanTrajectoryStep] = []
    step_index = 0
    for room in scene.rooms:
        x = room.origin[0] + stride
        limit = room.origin[0] + room.size[0] - stride
        while x < limit:
            steps.append(
                ScanTrajectoryStep(
                    step_id=f"move_{step_index:03d}",
                    action="move_forward",
                    position=[round(x, 2), round(room.origin[1] + room.size[1] / 2, 2), 1.6],
                    yaw_deg=0.0,
                    note=f"Sweep through room {room.room_id}.",
                )
            )
            x += stride
            step_index += 1
    return steps


def _mock_frames(scene: SceneGraph, steps: List[ScanTrajectoryStep], fps: int) -> List[VideoFrame]:
    visible_object_ids = [obj.object_id for room in scene.rooms for obj in room.objects]
    frames: List[VideoFrame] = []
    for index, step in enumerate(steps):
        frames.append(
            VideoFrame(
                frame_id=f"frame_{index:04d}",
                timestamp_sec=round(index / max(fps, 1), 2),
                visible_object_ids=visible_object_ids,
                observations={object_id: {"confidence": 1.0} for object_id in visible_object_ids},
            )
        )
    return frames


def build_scan_sequence(
    scene: SceneGraph,
    scan_id: str = "scan_000",
    orbit_radius: float = 1.0,
    orbit_views: int = 8,
    navigation_stride: float = 1.5,
    fps: int = 3,
) -> ScanSequence:
    steps = _build_orbit_steps(scene, radius=orbit_radius, orbit_views=orbit_views)
    steps.extend(_build_navigation_steps(scene, stride=navigation_stride))
    frames = _mock_frames(scene, steps, fps=fps)
    return ScanSequence(
        scene_id=scene.scene_id,
        scan_id=scan_id,
        steps=steps,
        frames=frames,
        metadata={
            "orbit_radius_m": orbit_radius,
            "orbit_views": orbit_views,
            "navigation_stride_m": navigation_stride,
            "fps": fps,
        },
    )


def export_blender_camera_script(scan: ScanSequence, output_path: Path) -> None:
    lines = [
        "import bpy",
        "from math import radians",
        "",
        "scene = bpy.context.scene",
        "for obj in list(bpy.data.objects):",
        "    if obj.type == 'CAMERA':",
        "        bpy.data.objects.remove(obj, do_unlink=True)",
        "bpy.ops.object.camera_add(location=(0.0, 0.0, 1.6))",
        "cam = bpy.context.object",
        "cam.name = 'ScanForgeCamera'",
        "cam.data.lens = 18",
        "cam.data.clip_start = 0.1",
        "cam.data.clip_end = 50.0",
        "",
    ]
    frame_cursor = 1
    for step in scan.steps:
        x, y, z = step.position
        lines.extend(
            [
                f"cam.location = ({x}, {y}, {z})",
                f"cam.rotation_euler = (radians(90), 0, radians({step.yaw_deg}))",
                f"cam.keyframe_insert(data_path='location', frame={frame_cursor})",
                f"cam.keyframe_insert(data_path='rotation_euler', frame={frame_cursor})",
                "",
            ]
        )
        frame_cursor += 1
    lines.extend(
        [
            "scene.frame_start = 1",
            f"scene.frame_end = {max(frame_cursor - 1, 1)}",
            "scene.render.image_settings.file_format = 'FFMPEG'",
            "scene.render.filepath = 'scanforgeqa_scan.mp4'",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_scan_from_scene_file(scene: SceneGraph, output_path: Path | None = None) -> ScanSequence:
    scan = build_scan_sequence(scene)
    if output_path is not None:
        dump_json(output_path, scan.to_dict())
    return scan

