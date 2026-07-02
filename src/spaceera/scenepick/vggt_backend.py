from __future__ import annotations

from contextlib import nullcontext
from typing import List, Mapping, Sequence

import numpy as np

from spaceera.scenepick.grounding import frame_semantic_mask
from spaceera.scenepick.sampling import FrameGeometry


class VGGTReconstructor:
    """Optional VGGT adapter. Heavy dependencies are imported only when used."""

    def __init__(self, model_path: str, device: str | None = None) -> None:
        try:
            import torch
            from vggt.models.vggt import VGGT
        except ImportError as exc:  # pragma: no cover - depends on optional software
            raise RuntimeError(
                "ScenePick reconstruction requires PyTorch and VGGT; see README.md"
            ) from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VGGT.from_pretrained(model_path).eval().to(self.device)

    def reconstruct(
        self,
        frame_paths: Sequence[str],
        grounding_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    ) -> List[FrameGeometry]:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        torch = self.torch
        images = load_and_preprocess_images(list(frame_paths)).to(self.device)[None]
        if self.device.startswith("cuda"):
            major = torch.cuda.get_device_capability()[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            autocast = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            autocast = nullcontext()

        with torch.no_grad(), autocast:
            tokens, patch_start_index = self.model.aggregator(images)
            pose = self.model.camera_head(tokens)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose, images.shape[-2:])
            depth, depth_confidence = self.model.depth_head(
                tokens, images, patch_start_index
            )

        points = unproject_depth_map_to_point_map(
            depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
        )
        confidence = depth_confidence.squeeze(0).detach().float().cpu().numpy()
        height, width = points.shape[1:3]
        geometry: List[FrameGeometry] = []
        for index, frame_path in enumerate(frame_paths):
            mask = None
            if grounding_boxes is not None:
                mask = frame_semantic_mask(frame_path, (height, width), grounding_boxes)
            geometry.append(
                FrameGeometry(
                    frame_path=frame_path,
                    points=np.asarray(points[index]),
                    confidence=np.asarray(confidence[index]),
                    semantic_mask=mask,
                )
            )
        return geometry
