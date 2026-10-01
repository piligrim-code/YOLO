# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2



def _run_in_batches(f, data_dict, out, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    data_len = len(out)
    if any(len(value) != data_len for value in data_dict.values()):
        raise ValueError("Batch input/output lengths differ")
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.asarray(bbox, dtype=np.float64).copy()
    if bbox.shape != (4,) or not np.isfinite(bbox).all() or np.any(bbox[2:] <= 0):
        return None
    if patch_shape is not None:
        if len(patch_shape) != 2 or any(int(v) != v or v <= 0 for v in patch_shape):
            raise ValueError("patch_shape must contain two positive integers")
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = np.r_[np.floor(bbox[:2]), np.ceil(bbox[2:])]

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]), bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox.astype(int)
    image = image[sy:ey, sx:ex]
    if patch_shape is not None:
        image = cv2.resize(image, tuple(int(v) for v in patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        import tensorflow as tf

        self.graph = tf.Graph()
        with tf.io.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())
        with self.graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name="net")
        self.input_var = self.graph.get_tensor_by_name("net/%s:0" % input_name)
        self.output_var = self.graph.get_tensor_by_name("net/%s:0" % output_name)
        if self.output_var.shape.rank != 2 or self.input_var.shape.rank != 4:
            raise ValueError("Encoder expects rank-4 images and rank-2 features")
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]
        if not self.feature_dim or any(v is None or v <= 0 for v in self.image_shape):
            raise ValueError("Encoder requires fixed positive image/feature dimensions")
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)

    def __call__(self, data_x, batch_size=32):
        if self.session is None:
            raise RuntimeError("Encoder is closed")
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                raise ValueError("Cannot encode an empty or out-of-frame box")
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    encoder.close = image_encoder.close
    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(int)
        min_frame_idx = frame_indices.min()
        max_frame_idx = frame_indices.max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    try:
        generate_detections(encoder, args.mot_dir, args.output_dir,
                            args.detection_dir)
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
