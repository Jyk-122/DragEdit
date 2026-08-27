"""Point-prompt SAM provider used by the local demo."""

import numpy as np


class SamMaskProvider:
    def __init__(self, checkpoint=None, model_type="vit_b", device=None):
        self.predictor = None
        if checkpoint:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = sam_model_registry[model_type](checkpoint=checkpoint).to(device)
            self.predictor = SamPredictor(model)

    @property
    def enabled(self):
        return self.predictor is not None

    def set_image(self, image):
        """Run SAM's image encoder once after a new image is loaded."""
        if self.enabled:
            self.predictor.set_image(image)

    def select(self, x, y):
        """Return the highest-scoring mask for one positive image-space point."""
        masks, scores, _ = self.predictor.predict(
            point_coords=np.array([[x, y]], np.float32),
            point_labels=np.array([1], np.int32),
            multimask_output=True,
        )
        return masks[scores.argmax()].astype(np.uint8)
