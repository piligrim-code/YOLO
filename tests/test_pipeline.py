import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import cv2
import numpy as np
import pytest

import main
from deepsort_tracker import Tracker
from video_pipeline import process_video


def result():
    return SimpleNamespace(boxes=SimpleNamespace(data=np.array([[5, 5, 30, 40, .9, 0]])))


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    state = SimpleNamespace(frames=[np.zeros((60, 80, 3), dtype=np.uint8) for _ in range(5)],
                            fps=1., advertised=5, capture_open=True, writer_open=True,
                            released=[], written=0)
    source, output = tmp_path / 'input.avi', tmp_path / 'output.avi'
    source.write_bytes(b'synthetic')
    class Capture:
        def __init__(self, path):
            self.frames = iter(state.frames)
        def isOpened(self):
            return state.capture_open
        def get(self, prop):
            return state.fps if prop == cv2.CAP_PROP_FPS else state.advertised
        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None
        def release(self):
            state.released.append('capture')
    class Writer:
        def __init__(self, path, *args):
            self.path = Path(path)
        def isOpened(self):
            return state.writer_open
        def write(self, frame):
            self.path.write_bytes(b'synthetic encoded frames')
            state.written += 1
        def release(self):
            state.released.append('writer')
    monkeypatch.setattr(cv2, 'VideoCapture', Capture)
    monkeypatch.setattr(cv2, 'VideoWriter', Writer)
    return state, source, output


def tracker():
    return Tracker(encoder=lambda frame, boxes: np.ones((len(boxes), 8), dtype=np.float32), n_init=2)


def test_one_tracker_update_per_frame_and_optional_alerts(fake_video):
    state, source, output = fake_video
    calls, alerts = [], []
    tr = tracker()
    original = tr.update
    def update(frame, rows):
        calls.append(rows)
        original(frame, rows)
    tr.update = update
    async def alert(frame, track_id):
        alerts.append(track_id)
    stats = asyncio.run(process_video(source, output, lambda *a, **k: [result()], tr,
                                      alert_after=2, send_alert=alert))
    assert stats == {'frames': 5, 'alerts': 1}
    assert len(calls) == 5 and state.written == 5 and len(alerts) == 1
    assert state.released == ['writer', 'capture']


def test_empty_detector_results_still_age_tracker(fake_video):
    state, source, output = fake_video
    tr = tracker()
    calls = []
    tr.update = lambda frame, rows: calls.append(rows)
    assert asyncio.run(process_video(source, output, lambda *a, **k: [], tr))['frames'] == 5
    assert calls == [[]] * 5


@pytest.mark.parametrize('fps', [0, -1, float('nan'), float('inf')])
def test_invalid_fps_releases_capture_without_output(fake_video, fps):
    state, source, output = fake_video
    state.fps = fps
    with pytest.raises(ValueError):
        asyncio.run(process_video(source, output, lambda *a, **k: [], tracker()))
    assert state.released == ['capture']
    assert not output.exists()


def test_explicit_fps_override(fake_video):
    state, source, output = fake_video
    state.fps = 0
    assert asyncio.run(process_video(source, output, lambda *a, **k: [], tracker(), fps_override=25))['frames'] == 5


@pytest.mark.parametrize('kind', ['capture', 'empty', 'writer', 'dimension', 'early_eof', 'inference', 'alert'])
def test_failures_release_and_remove_owned_output(fake_video, kind):
    state, source, output = fake_video
    if kind == 'capture': state.capture_open = False
    if kind == 'empty': state.frames = []
    if kind == 'writer': state.writer_open = False
    if kind == 'dimension': state.frames[2] = np.zeros((10, 10, 3), dtype=np.uint8)
    if kind == 'early_eof': state.advertised = 9
    def detect(*args, **kwargs):
        if kind == 'inference': raise RuntimeError('synthetic inference failure')
        return [result()]
    async def alert(*args):
        raise RuntimeError('synthetic alert failure')
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(process_video(source, output, detect, tracker(), alert_after=1,
                                  send_alert=alert if kind == 'alert' else None))
    assert 'capture' in state.released
    if kind not in ('capture', 'empty'):
        assert 'writer' in state.released
    assert not output.exists()


def test_cancellation_cleans_resources(fake_video):
    state, source, output = fake_video
    async def alert(*args):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(process_video(source, output, lambda *a, **k: [result()], tracker(),
                                  alert_after=1, send_alert=alert))
    assert state.released == ['writer', 'capture']
    assert not output.exists()


def test_preexisting_output_not_overwritten_or_deleted(fake_video):
    state, source, output = fake_video
    output.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        asyncio.run(process_video(source, output, lambda *a, **k: [], tracker()))
    assert output.read_bytes() == b'keep'
    assert state.released == ['capture']


def test_same_input_output_refused(fake_video):
    state, source, output = fake_video
    with pytest.raises(ValueError):
        asyncio.run(process_video(source, source, lambda *a, **k: [], tracker()))
    assert source.read_bytes() == b'synthetic'
    assert state.released == []


def test_real_opencv_video_and_real_tracker(tmp_path):
    source, output = tmp_path / 'synthetic.avi', tmp_path / 'result.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 5., (80, 60))
    assert writer.isOpened(), 'MJPG codec required for this regression'
    try:
        for i in range(10):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            frame[10:35, 10 + i:25 + i] = (0, 80, 200)
            writer.write(frame)
    finally:
        writer.release()
    stats = asyncio.run(process_video(source, output, lambda *a, **k: [result()], tracker()))
    assert stats == {'frames': 10, 'alerts': 0}
    capture = cv2.VideoCapture(str(output))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            count += 1
            assert frame.shape == (60, 80, 3)
    finally:
        capture.release()
    assert count == 10


@pytest.mark.parametrize('failure', ['model', 'classes', 'pipeline', None])
@pytest.mark.parametrize('enabled', [False, True])
def test_run_closes_encoder_and_bot(monkeypatch, tmp_path, failure, enabled):
    events = []
    class Adapter:
        def __init__(self, path): events.append('tracker')
        def close(self): events.append('close tracker')
    class Bot:
        def __init__(self, **kwargs): self.session = self
        async def close(self): events.append('close bot')
    module = ModuleType('ultralytics')
    def detector(path):
        if failure == 'model': raise RuntimeError('synthetic model failure')
        return SimpleNamespace(names={0: 'car' if failure == 'classes' else 'person'})
    module.YOLO = detector
    monkeypatch.setitem(sys.modules, 'ultralytics', module)
    import deepsort_tracker, aiogram
    monkeypatch.setattr(deepsort_tracker, 'Tracker', Adapter)
    monkeypatch.setattr(aiogram, 'Bot', Bot)
    monkeypatch.setenv('TOKEN', 'synthetic')
    monkeypatch.setenv('CHAT_ID', 'synthetic')
    async def pipeline(*args, **kwargs):
        if failure == 'pipeline': raise RuntimeError('synthetic pipeline failure')
        return {'frames': 1, 'alerts': 0}
    monkeypatch.setattr(main, 'process_video', pipeline)
    args = SimpleNamespace(encoder=tmp_path / 'encoder.pb', weights=tmp_path / 'model.pt',
                           input=tmp_path / 'in.avi', output=tmp_path / 'out.avi',
                           telegram=enabled, confidence=.5, fps=None, alert_after=3)
    if failure:
        with pytest.raises((RuntimeError, ValueError)): asyncio.run(main.run(args))
    else:
        assert asyncio.run(main.run(args))['frames'] == 1
    assert events[-1] == 'close tracker'
    assert ('close bot' in events) == (enabled and failure not in ('model', 'classes'))
