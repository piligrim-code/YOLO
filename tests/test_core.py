import asyncio
import builtins
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import main
from deepsort_tracker import Tracker
from deep_sort.tools.generate_detections import extract_image_patch, _run_in_batches
from video_pipeline import PresenceClock, detection_rows


def result(rows):
    return SimpleNamespace(boxes=SimpleNamespace(data=np.asarray(rows)))


def test_import_does_not_construct_services(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('cv2', 'ultralytics', 'tensorflow', 'aiogram'):
            pytest.fail('Unexpected provider import')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    importlib.reload(main)


def test_original_coordinate_regression_and_person_filter():
    rows = [[1.25, 2.5, 20.75, 30.5, .9, 0], [1, 2, 20, 30, .9, 2]]
    assert detection_rows([result(rows)], 80, 60, .5) == [[1.25, 2.5, 20.75, 30.5, .9]]


def test_clipping_and_invalid_detections():
    rows = [[-4, -2, 200, 80, .5, 0], [2, 2, 1, 3, .9, 0],
            [0, 0, 20, 20, math.nan, 0], [0, 0, 20, 20, 1.1, 0],
            [0, 0, 20, 20, .4, 0], [0, 0, 20, 20, .9, math.inf]]
    assert detection_rows([result(rows)], 80, 60, .5) == [[0., 0., 80, 60, .5]]
    assert detection_rows([SimpleNamespace(boxes=None)], 80, 60, .5) == []


def test_malformed_detection_row_fails():
    with pytest.raises(ValueError):
        detection_rows([result([[1, 2, 3]])], 80, 60, .5)


def test_presence_one_alert_per_contiguous_observation():
    clock = PresenceClock(3)
    assert clock.update([1, 1], 0) == []
    assert clock.update([1], 2.9) == []
    assert clock.update([1], 3) == [1]
    assert clock.update([1], 20) == []
    assert clock.update([], 21) == []
    assert clock.active == {}
    assert clock.update([1], 22) == []
    assert clock.update([1], 25) == [1]


@pytest.mark.parametrize('value', [0, -1, math.inf, math.nan])
def test_invalid_alert_interval(value):
    with pytest.raises(ValueError):
        PresenceClock(value)


def test_presence_time_must_be_monotonic():
    clock = PresenceClock(1)
    clock.update([1], 2)
    for bad in (1, math.inf, math.nan):
        with pytest.raises(ValueError):
            clock.update([1], bad)


def test_real_deepsort_tracks_synthetic_boxes_and_missed_frame():
    calls = []
    def encoder(frame, boxes):
        calls.append(boxes.copy())
        return np.ones((len(boxes), 8), dtype=np.float32)
    tracker = Tracker(encoder=encoder, n_init=2, max_age=2)
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    tracker.update(frame, [[5., 6., 25., 36., .9]])
    assert tracker.tracks == []
    tracker.update(frame, [[6., 6., 26., 36., .9]])
    assert len(tracker.tracks) == 1
    track_id = tracker.tracks[0].track_id
    np.testing.assert_allclose(calls[0], [[5, 6, 20, 30]])
    tracker.update(frame, [])
    assert tracker.tracks == []  # Predictions alone never count as presence.
    tracker.update(frame, [[7., 6., 27., 36., .9]])
    assert tracker.tracks[0].track_id == track_id
    for _ in range(3):
        tracker.update(frame, [])
    assert tracker.tracker.tracks == []
    tracker.close()
    with pytest.raises(RuntimeError):
        tracker.update(frame, [])


@pytest.mark.parametrize('detections', [[[1, 2, 1, 4, .9]], [[1, 2, 3, 4, 2]],
                                       [[1, 2, 3, 4, math.nan]], [[1, 2, 3]]])
def test_bad_tracker_input_never_reaches_encoder(detections):
    tracker = Tracker(encoder=lambda *args: pytest.fail('Invalid boxes reached encoder'))
    with pytest.raises(ValueError):
        tracker.update(np.zeros((10, 10, 3)), detections)
    assert tracker.tracker.tracks == []


@pytest.mark.parametrize('features', [np.zeros((1, 8)), np.ones((2, 8)), np.full((1, 8), math.nan)])
def test_invalid_encoder_features(features):
    tracker = Tracker(encoder=lambda *args: features)
    with pytest.raises(ValueError):
        tracker.update(np.zeros((10, 10, 3)), [[1, 1, 4, 5, .9]])


def test_large_feature_values_do_not_overflow_cosine_distance():
    tracker = Tracker(encoder=lambda *a: np.full((1, 8), 1e30, dtype=np.float32), n_init=2)
    with np.errstate(all='raise'):
        for _ in range(3):
            tracker.update(np.zeros((10, 10, 3)), [[1, 1, 4, 5, .9]])
    assert len(tracker.tracks) == 1


def test_encoder_close_is_idempotent():
    events = []
    encoder = SimpleNamespace(close=lambda: events.append('close'))
    tracker = Tracker(encoder=encoder)
    tracker.close()
    tracker.close()
    assert events == ['close']


@pytest.mark.parametrize('value', [0, -1, math.nan, 1.5, True])
def test_tracker_configuration(value):
    with pytest.raises(ValueError):
        Tracker(encoder=lambda *args: None, max_age=value)


def test_missing_model_does_not_import_tensorflow(tmp_path):
    with pytest.raises(ValueError):
        Tracker(tmp_path / 'missing.pb')


def test_patch_includes_last_pixel_and_handles_none_shape():
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    patch = extract_image_patch(frame, [3, 3, 1, 1], None)
    np.testing.assert_array_equal(patch, frame[3:4, 3:4])
    assert extract_image_patch(frame, [20, 20, 2, 2], (4, 2)) is None
    assert extract_image_patch(frame, [0, 0, 0, 2], (4, 2)) is None
    assert extract_image_patch(frame, [0, 0, math.nan, 2], (4, 2)) is None


def test_integer_box_aspect_correction_uses_float_copy():
    frame = np.ones((20, 20, 3), dtype=np.uint8)
    box = np.array([5, 5, 4, 5])
    patch = extract_image_patch(frame, box, (8, 3))
    assert patch.shape == (8, 3, 3)
    np.testing.assert_array_equal(box, [5, 5, 4, 5])


def test_batching_with_remainder_and_empty_batch():
    values = np.arange(5).reshape(5, 1)
    out = np.zeros_like(values)
    _run_in_batches(lambda d: d['x'] * 2, {'x': values}, out, 2)
    np.testing.assert_array_equal(out, values * 2)
    _run_in_batches(lambda d: pytest.fail('Empty encoder call'), {'x': []}, np.zeros((0, 1)), 2)


@pytest.mark.parametrize('batch_size', [0, -1, 1.5, True])
def test_batch_size_validation(batch_size):
    with pytest.raises(ValueError):
        _run_in_batches(lambda d: d, {'x': []}, np.zeros((0, 1)), batch_size)


def cli_files(tmp_path):
    paths = [tmp_path / 'in.avi', tmp_path / 'model.pt', tmp_path / 'encoder.pb']
    for path in paths:
        path.touch()
    return ['--input', str(paths[0]), '--output', str(tmp_path / 'out.avi'),
            '--weights', str(paths[1]), '--encoder', str(paths[2])]


def test_cli_validates_before_services(tmp_path, monkeypatch):
    monkeypatch.delenv('TOKEN', raising=False)
    monkeypatch.delenv('CHAT_ID', raising=False)
    argv = cli_files(tmp_path)
    assert main.arguments(argv).telegram is False
    with pytest.raises(SystemExit) as error:
        main.arguments(argv + ['--telegram'])
    assert error.value.code == 2
    (tmp_path / 'out.avi').write_bytes(b'keep')
    with pytest.raises(SystemExit):
        main.arguments(argv)


@pytest.mark.parametrize('flag,value', [('--fps', '0'), ('--fps', 'nan'),
                                       ('--confidence', 'nan'), ('--confidence', '2')])
def test_cli_invalid_numeric_values(tmp_path, flag, value):
    with pytest.raises(SystemExit):
        main.arguments(cli_files(tmp_path) + [flag, value])


def test_main_errors_do_not_echo_provider_secrets(tmp_path, monkeypatch, capsys):
    async def fail(args):
        raise RuntimeError('synthetic-private-value')
    monkeypatch.setattr(main, 'run', fail)
    assert main.main(cli_files(tmp_path)) == 1
    output = capsys.readouterr().err
    assert 'RuntimeError' in output
    assert 'synthetic-private-value' not in output


def test_telegram_uses_buffered_input_file_without_network():
    from aiogram.types import BufferedInputFile
    calls = []
    class Bot:
        async def send_photo(self, **kwargs):
            calls.append(kwargs)
    send = main.telegram_sender(Bot(), 'synthetic-chat')
    asyncio.run(send(np.zeros((16, 16, 3), dtype=np.uint8), 7))
    assert len(calls) == 1
    assert isinstance(calls[0]['photo'], BufferedInputFile)
    assert calls[0]['photo'].data[:2] == b'\xff\xd8'
    assert calls[0]['request_timeout'] == 20
