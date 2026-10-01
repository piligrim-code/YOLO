# Local Person Tracking Prototype

Ultralytics detections plus the vendored DeepSORT tracker, with optional
Telegram frame alerts. This revision repairs the historical startup and
tracking flow; it is not a production-qualified surveillance system or a
measured detection/re-identification benchmark. ByteTrack is not implemented.

## Offline demo and checks

Python 3.12 is the CI target. Use an isolated environment:

```console
python -m pip install -r requirements-test.txt
python -m pytest tests --ignore=tests/test_tensorflow_encoder.py -q
python main.py --help
python demo.py
python demo.py --output synthetic-demo.avi
```

The demo generates 32 artificial frames, supplies scripted boxes and constant
appearance features, then runs the real DeepSORT and OpenCV pipeline. It
produces one final track and one local alert callback. No camera, real footage,
trained model, TensorFlow session or Telegram client is used. Those numbers
are a deterministic regression result, not detection accuracy. The optional
output lets you inspect the generated video; existing files are never replaced.

Core CI runs on Linux and Windows. A separate CPU-only TensorFlow job creates
a tiny artificial graph and checks tensor namespaces, graph isolation, batch
processing, edge crops and session closure. It does not use MARS weights.

```console
python -m pip install "tensorflow>=2.16,<3"
python -m pytest tests/test_tensorflow_encoder.py -q
```

## Local video with trusted model files

The real model path additionally requires the runtime dependencies and
separately obtained detector/appearance weights:

```console
python -m pip install -r requirements-runtime.txt
python main.py --input reviewed-input.avi --output tracked-output.avi --weights models/yolov8n.pt --encoder models/mars-small128.pb
```

Input must be an existing local file. Camera indices and network stream URLs
are not accepted. Both weight files must exist locally; this application does
not deliberately download them. Only use trusted weights with verified usage
terms. Ultralytics/TensorFlow are not network- or resource-sandboxed by this CLI.
The full pretrained runtime and its GPU compatibility still require a separate
integration run. Requirement ranges are not a fully locked deployment stack.

The detector must expose COCO-compatible `person` at class 0. Other classes
are filtered out because this appearance encoder is person-oriented. The
application does not identify people, implement facial recognition or establish
that track IDs correspond to stable real-world identities.

Options:

- `--confidence 0.5`: minimum person detection score.
- `--fps 25`: explicit override when file metadata is missing/invalid.
- `--alert-after 3`: seconds of continuously observed, confirmed tracking.
- `--telegram`: explicit opt-in to uploading frames; disabled by default.

Timing uses frame index divided by constant source FPS, not inference speed.
Variable-frame-rate footage is not timestamp-accurate with this implementation.
A missed detection resets presence timing; predictions alone do not count.
One alert is emitted per uninterrupted observation period, not every frame.
Alert delivery is sequential and fail-fast; there is no retry/durable queue.

AVI uses MJPG; MP4 requests mp4v and requires codec support. Headless output
is the default. Capture/writer handles are closed on success, Python errors
and cancellation. A newly created partial output is removed on failure;
existing output is refused. Hard process termination and hostile concurrent
filesystem changes are outside that guarantee. Early EOF is detected when
usable frame-count metadata reports missing frames; OpenCV cannot always
distinguish corrupt decoding from EOF when that metadata is absent.

## Telegram and data handling

Only with `--telegram`, set `TOKEN` and `CHAT_ID` in your process environment.
No `.env` file is loaded automatically and no credentials are committed.
Never paste real credentials into source, shell transcripts or issues.

Enabling alerts sends the full annotated frame to the configured Telegram
chat, potentially including bystanders or private surroundings. Review input,
destination, access and retention before opting in. This prototype does not
implement consent, face blurring, access control or a retention policy.
No live Telegram request was made by the offline tests.

## Third-party source and remaining work

See `THIRD_PARTY.md` and `deep_sort/LICENSE` for the upstream DeepSORT source
reference and restored GPLv3 text. The surrounding application's licensing,
dependencies and model rights still require review; this is not a blanket MIT
release. The old TensorFlow 1 `freeze_model.py` is retained as historical source,
not part of the supported commands. The historical MOT batch-export helper
has not received full workflow qualification.

Before production use: validate genuine detector/encoder weights, GPU/runtime
compatibility, occlusion/crossing/multi-person tracking, codecs and real alert
delivery; evaluate quality on an authorized dataset; add operational/privacy
controls. Offline synthetic success does not establish any of these results.
