"""DeepSORT adapter; TensorFlow is loaded only when creating a real encoder."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray


class Tracker:
    def __init__(self, encoder_model_filename=None, *, encoder=None, n_init=8, max_age=100):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (n_init, max_age)):
            raise ValueError('n_init and max_age must be positive')
        metric = nn_matching.NearestNeighborDistanceMetric('cosine', 0.4, 100)
        self.tracker = DeepSortTracker(metric, max_iou_distance=0.7, max_age=max_age, n_init=n_init)
        if encoder is None:
            if encoder_model_filename is None or not Path(encoder_model_filename).is_file():
                raise ValueError('A trusted local encoder graph is required')
            from deep_sort.tools.generate_detections import create_box_encoder
            encoder = create_box_encoder(str(encoder_model_filename), batch_size=1)
        self.encoder = encoder
        self.tracks = []
        self.closed = False

    def update(self, frame, detections):
        if self.closed:
            raise RuntimeError('Tracker is closed')
        dets = []
        if len(detections):
            values = np.asarray(detections, dtype=np.float32)
            if (values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all()
                    or (values[:, 2:4] <= values[:, :2]).any()
                    or (values[:, 4] < 0).any() or (values[:, 4] > 1).any()):
                raise ValueError('Invalid tracking detections')
            boxes = values[:, :4].copy()
            boxes[:, 2:] -= boxes[:, :2]
            features = np.asarray(self.encoder(frame, boxes), dtype=np.float32)
            if (features.ndim != 2 or features.shape[0] != len(boxes) or features.shape[1] == 0
                    or not np.isfinite(features).all()):
                raise ValueError('Encoder must return finite nonzero feature vectors')
            norms = np.linalg.norm(features.astype(np.float64), axis=1)
            if (norms == 0).any():
                raise ValueError('Encoder must return nonzero feature vectors')
            features = (features / norms[:, None]).astype(np.float32)
            dets = [Detection(box, score, feature)
                    for box, score, feature in zip(boxes, values[:, 4], features)]
        self.tracker.predict()
        self.tracker.update(dets)
        self.tracks = [Track(t.track_id, t.to_tlbr()) for t in self.tracker.tracks
                       if t.is_confirmed() and t.time_since_update == 0]

    def close(self):
        if not self.closed:
            self.closed = True
            close = getattr(self.encoder, 'close', None)
            if close is not None:
                close()
