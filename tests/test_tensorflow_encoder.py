"""Optional real TensorFlow checks using a newly generated, trivial graph."""
import numpy as np
import pytest

tf = pytest.importorskip('tensorflow')
from deep_sort.tools.generate_detections import ImageEncoder, create_box_encoder


def graph_file(tmp_path, wrong_rank=False):
    graph = tf.Graph()
    with graph.as_default():
        images = tf.compat.v1.placeholder(tf.uint8, [None, 8, 4, 3], name='images')
        if wrong_rank:
            tf.identity(images, name='features')
        else:
            tf.reduce_mean(tf.cast(images, tf.float32), axis=[1, 2], name='features')
    path = tmp_path / 'synthetic.pb'
    path.write_bytes(graph.as_graph_def().SerializeToString())
    return path


def test_real_graph_namespace_isolation_batching_and_close(tmp_path):
    path = graph_file(tmp_path)
    eager_before = tf.executing_eagerly()
    default = tf.compat.v1.get_default_graph()
    before = len(default.get_operations())
    first, second = ImageEncoder(str(path)), ImageEncoder(str(path))
    try:
        assert first.graph is not second.graph
        assert first.input_var.name == 'net/images:0'
        assert first.output_var.name == 'net/features:0'
        values = np.full((5, 8, 4, 3), 7, dtype=np.uint8)
        np.testing.assert_allclose(first(values, batch_size=2), np.full((5, 3), 7))
        np.testing.assert_allclose(second(values[:1]), [[7, 7, 7]])
        assert len(default.get_operations()) == before
        assert tf.executing_eagerly() == eager_before
    finally:
        first.close()
        second.close()
    first.close()
    with pytest.raises(RuntimeError):
        first(np.zeros((1, 8, 4, 3), dtype=np.uint8))


def test_box_encoder_with_actual_graph_and_border_crop(tmp_path):
    encoder = create_box_encoder(str(graph_file(tmp_path)), batch_size=1)
    try:
        frame = np.full((8, 4, 3), 20, dtype=np.uint8)
        features = encoder(frame, [[0, 0, 4, 8], [3, 7, 1, 1]])
        np.testing.assert_allclose(features, [[20, 20, 20], [20, 20, 20]])
        assert encoder(frame, []).shape == (0, 3)
        with pytest.raises(ValueError):
            encoder(frame, [[100, 100, 1, 1]])
    finally:
        encoder.close()


def test_invalid_graph_does_not_create_session(tmp_path, monkeypatch):
    path = graph_file(tmp_path, wrong_rank=True)
    monkeypatch.setattr(tf.compat.v1, 'Session', lambda **kwargs: pytest.fail('Session before validation'))
    with pytest.raises(ValueError):
        ImageEncoder(str(path))


def test_missing_tensor_does_not_create_session(tmp_path, monkeypatch):
    path = graph_file(tmp_path)
    monkeypatch.setattr(tf.compat.v1, 'Session', lambda **kwargs: pytest.fail('Session before validation'))
    with pytest.raises(KeyError):
        ImageEncoder(str(path), input_name='missing')
