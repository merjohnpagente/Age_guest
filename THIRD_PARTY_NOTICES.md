# Third-party notices

FACELENS bundles the following third-party software and model files so that the demo can run entirely in the browser without external model CDNs.

## MediaPipe Tasks Vision

- Project: MediaPipe
- Owner: Google LLC
- License: Apache License 2.0
- Source: <https://github.com/google-ai-edge/mediapipe>
- Files used:
  - `static/vendor/vision_bundle.mjs`
  - `static/vendor/wasm/vision_wasm_internal.js`
  - `static/vendor/wasm/vision_wasm_internal.wasm`
  - `static/vendor/wasm/vision_wasm_nosimd_internal.js`
  - `static/vendor/wasm/vision_wasm_nosimd_internal.wasm`
- Model: BlazeFace short-range face detector (`static/models/blaze_face_short_range.tflite`)
- Model source: MediaPipe face detector model card / model assets
- License: Apache License 2.0

## face-api.js

- Project: face-api.js
- Original author: Vincent Mühler (justadudewhohacks)
- Maintained fork used here: Vladmandic
- License: MIT
- Sources:
  - <https://github.com/vladmandic/face-api>
  - <https://github.com/justadudewhohacks/face-api.js>
- Files used:
  - `static/vendor/faceapi/face-api.js`
  - `static/models/faceapi/age_gender_model-weights_manifest.json`
  - `static/models/faceapi/age_gender_model.bin`
  - `static/models/faceapi/face_expression_model-weights_manifest.json`
  - `static/models/faceapi/face_expression_model.bin`

The age/gender and expression model weights are redistributed from the face-api.js model repository. They were published for research and demonstration purposes. The models are not trained or validated by this project, and their outputs are probabilistic estimates rather than facts.

## TensorFlow.js

- Project: TensorFlow.js
- Owner: Google LLC
- License: Apache License 2.0
- Source: <https://github.com/tensorflow/tfjs>
- The face-api.js bundle includes TensorFlow.js code.

## Datasets and model limitations

The bundled age/gender and expression models were trained by their original authors on public face datasets. This project does not redistribute those datasets. Predictions can reflect the limitations and biases of the training data, including bias across age, gender presentation, skin tone, lighting, and camera hardware.

The models are included only to demonstrate on-device inference. They are not suitable for identity verification, surveillance, profiling, age verification, or any high-impact decision.

## App code

The FACELENS interface, tracking logic, and integration code in this repository are original project code. The project is provided as a demonstration and without warranty; see the repository README for responsible-use guidance.
