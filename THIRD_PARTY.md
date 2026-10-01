# Third-Party Source And Model Boundaries

## DeepSORT

`deep_sort/deep_sort/*.py` and `deep_sort/tools/*.py` are derived from
`nwojke/deep_sort`, associated with Nicolai Wojke, Alex Bewley and collaborators.
The upstream repository was inspected at commit
`f08cf1dc470eeb1cd2add1cbf077d95ac6c48aab`.

Before this correction, six of the ten Python files matched that upstream
snapshot after line-ending normalization; detection, assignment and the two
tools already contained local modifications. This comparison identifies a
source reference, not the exact historical vendoring date.

The upstream GNU GPL version 3 license text is retained in `deep_sort/LICENSE`.
Existing author/source headers remain. This patch does not relicense the
vendored code or claim that the surrounding application has an MIT license.
The root application had no license file; its distribution terms still need
an explicit owner review together with all dependency/model terms.

The correction dated 2026-10-01 modifies `tools/generate_detections.py`: lazy
TensorFlow import, isolated and correctly named graph tensors, deterministic
invalid-crop failures, boundary-inclusive crops, batch validation and explicit
session closure. The old `freeze_model.py` remains historical TensorFlow 1
source and is not part of the supported runtime or test commands.

## Detector And Encoder Assets

Ultralytics, TensorFlow, OpenCV, NumPy, SciPy and aiogram retain their own
terms. Review the selected versions and model assets before redistribution.
No pretrained detector/appearance weights are bundled or fetched by the demo.
`mars-small128.pb` must be obtained separately with its provenance and usage
terms verified. The appearance encoder is person-oriented, so the application
filters Ultralytics COCO detections to class 0 (person).

Only load trusted local `.pt` and `.pb` assets. This application is not a
sandbox for untrusted models. Restoring a source license is not qualification
of model rights, commercial suitability or full dependency compatibility.
