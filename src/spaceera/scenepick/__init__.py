"""ScenePick frame sampling for the SpaceEra++ journal extension."""

from spaceera.scenepick.sampling import (
    FrameGeometry,
    SelectionResult,
    build_weighted_voxel_sets,
    greedy_maximum_coverage,
    select_frames,
    uniform_sample_paths,
)

__all__ = [
    "FrameGeometry",
    "SelectionResult",
    "build_weighted_voxel_sets",
    "greedy_maximum_coverage",
    "select_frames",
    "uniform_sample_paths",
]
