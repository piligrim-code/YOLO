"""Synthetic video and scripted detections; no trained models or external services."""
import argparse
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def run_demo(output=None):
    import cv2
    import numpy as np
    from deepsort_tracker import Tracker
    from video_pipeline import process_video

    with TemporaryDirectory(prefix='synthetic-tracking-') as directory:
        source = Path(directory) / 'input.avi'
        destination = Path(output) if output else Path(directory) / 'output.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 8., (80, 60))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError('MJPG writer unavailable')
        try:
            for i in range(32):
                frame = np.zeros((60, 80, 3), dtype=np.uint8)
                frame[10:40, 5 + i:25 + i] = (40, 120, 220)
                writer.write(frame)
        finally:
            writer.release()
        index = 0
        def detector(frame, **kwargs):
            nonlocal index
            row = np.array([[5 + index, 10, 25 + index, 40, .95, 0]], dtype=float)
            index += 1
            return [SimpleNamespace(boxes=SimpleNamespace(data=row))]
        tracker = Tracker(encoder=lambda frame, boxes: np.ones((len(boxes), 8), dtype=np.float32), n_init=2)
        events = []
        async def local_event(frame, track_id):
            events.append(track_id)
        try:
            stats = asyncio.run(process_video(source, destination, detector, tracker, send_alert=local_event))
            ids = [track.track_id for track in tracker.tracks]
        finally:
            tracker.close()
        return {'synthetic': True, **stats, 'final_track_ids': ids,
                'local_alert_events': len(events), 'alert_delivery': 'local_callback'}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, help='Optional new .avi or .mp4 file to keep')
    print(json.dumps(run_demo(cli.parse_args().output)))
