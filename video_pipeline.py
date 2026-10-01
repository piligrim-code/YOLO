"""Local-video pipeline with deterministic source-time alerts and owned output."""
from contextlib import ExitStack
import math
from pathlib import Path


def detection_rows(results, width, height, confidence):
    rows = []
    for result in results:
        if result.boxes is None:
            continue
        for raw in result.boxes.data.tolist():
            if len(raw) != 6:
                raise ValueError('Expected detector rows: x1,y1,x2,y2,score,class_id')
            x1, y1, x2, y2, score, class_id = map(float, raw)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score, class_id)):
                continue
            # The bundled MARS encoder is for people, not arbitrary object classes.
            if class_id != 0 or not 0 <= score <= 1 or score < confidence:
                continue
            x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
            y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
            if x2 > x1 and y2 > y1:
                rows.append([x1, y1, x2, y2, score])
    return rows


class PresenceClock:
    def __init__(self, seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Alert interval must be positive and finite')
        self.seconds = seconds
        self.active = {}
        self.last_time = -math.inf

    def update(self, ids, timestamp):
        if not math.isfinite(timestamp) or timestamp < self.last_time:
            raise ValueError('Source time must be finite and monotonic')
        self.last_time = timestamp
        ids = set(ids)
        self.active = {key: value for key, value in self.active.items() if key in ids}
        due = []
        for key in sorted(ids):
            first, sent = self.active.setdefault(key, (timestamp, False))
            if not sent and timestamp - first >= self.seconds:
                due.append(key)
                self.active[key] = (first, True)
        return due


async def process_video(input_path, output_path, detector, tracker, *, confidence=0.5,
                        fps_override=None, alert_after=3., send_alert=None):
    import cv2

    source, destination = Path(input_path), Path(output_path)
    if not source.is_file() or source.resolve() == destination.resolve():
        raise ValueError('Input must be a local file distinct from output')
    if destination.suffix.lower() not in ('.avi', '.mp4'):
        raise ValueError('Output must be .avi or .mp4')
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Confidence must be between zero and one')
    clock = PresenceClock(alert_after)
    owned_output = False
    try:
        with ExitStack() as resources:
            capture = cv2.VideoCapture(str(source))
            resources.callback(capture.release)
            if not capture.isOpened():
                raise ValueError('Cannot open input video')
            fps = fps_override if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError('Invalid video FPS; provide an explicit override')
            advertised_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError('Video contains no decodable frames')
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                raise ValueError('Empty frame')
            shape = frame.shape
            with destination.open('xb'):
                pass
            owned_output = True
            codec = 'MJPG' if destination.suffix.lower() == '.avi' else 'mp4v'
            writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
            resources.callback(writer.release)
            if not writer.isOpened():
                raise ValueError('Cannot open video writer')
            frames = alerts = 0
            while ok:
                if frame is None or frame.shape != shape:
                    raise ValueError('Video frame dimensions changed')
                detections = detection_rows(detector(frame, verbose=False), width, height, confidence)
                tracker.update(frame, detections)
                for track in tracker.tracks:
                    x1, y1, x2, y2 = track.bbox
                    cv2.rectangle(frame, (max(0, min(width - 1, int(x1))), max(0, min(height - 1, int(y1)))),
                                  (max(0, min(width - 1, int(x2))), max(0, min(height - 1, int(y2)))),
                                  (40, 180, 80), 2)
                due = clock.update((t.track_id for t in tracker.tracks), frames / fps)
                if send_alert is not None:
                    for track_id in due:
                        await send_alert(frame, track_id)
                        alerts += 1
                writer.write(frame)
                frames += 1
                ok, frame = capture.read()
            if math.isfinite(advertised_frames) and advertised_frames > frames + 0.5:
                raise ValueError('Video ended before its advertised frame count')
        if destination.stat().st_size == 0:
            raise ValueError('Video writer produced an empty file')
        return {'frames': frames, 'alerts': alerts}
    except BaseException:
        if owned_output:
            destination.unlink(missing_ok=True)
        raise
