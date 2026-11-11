# FACELENS

Live, private face analysis in the browser. Point a camera at one or more people and FACELENS draws a tracking box around each face, then shows an estimated age range, gender, expression, detection confidence, and a temporary Face ID.

**Live demo:** <https://merjohnpagente.github.io/Age_guest/>

All detection and inference run on the device. Camera frames are never uploaded.

## What it shows

- Red face boxes with smooth tracking and multi-face support
- **Age estimate** as a range
- **Gender** as Woman or Man, smoothed across predictions
- **Expression** from a small on-device classifier
- **Confidence** for face detection (not for age/gender/expression accuracy)
- **Face ID** that is temporary and anonymous
- Start/stop camera and fullscreen controls

Age, gender, and expression are experimental AI estimates. They can be wrong and must not be treated as facts or used to make decisions about a person.

## How it works

| Stage | Technology |
| --- | --- |
| Face detection | MediaPipe Tasks Vision FaceDetector with the BlazeFace short-range model |
| Age + gender estimate | face-api.js `ageGenderNet`, running locally through TensorFlow.js |
| Expression estimate | face-api.js `faceExpressionNet`, running locally through TensorFlow.js |
| Tracking | Lightweight IoU + proximity matching with a short re-identification memory so temporary Face IDs stay stable |

The MediaPipe runtime, WASM binaries, and all model weights are vendored into `static/`, so the live site does not depend on third-party model CDNs.

## Requirements

- A modern browser with WebGL and camera support
- Camera permission
- A secure context (`https://` or `http://localhost`) for camera access on phones and most browsers

Detection runs on the GPU when available and falls back to the CPU delegate if needed. Very old or locked-down devices may not support the required WebAssembly/WebGL features.

## Run locally

```bash
python3 serve.py --port 8000
```

Open <http://localhost:8000/>.

`serve.py` is only a small static file server. It exists to provide the correct MIME types for `.wasm`, `.mjs`, and `.tflite` files. The face analysis itself is entirely JavaScript in the browser.

You can also use any static server that serves the repository root, for example:

```bash
python3 -m http.server 8000
```

If the browser cannot load the MediaPipe WASM files, check that the server sends `application/wasm` for `.wasm` and a JavaScript MIME type for `.mjs`.

## Deploy

The repository deploys automatically to GitHub Pages through `.github/workflows/deploy-pages.yml` on every push to `main`.

The workflow:

1. Copies `index.html`, `static/`, and the notices into a `dist/` folder
2. Adds `.nojekyll`
3. Uploads the folder as a Pages artifact
4. Deploys it with GitHub's official Pages actions

No build step or server-side runtime is required.

## Privacy

- Camera frames are processed in the browser and are not sent to a server.
- The app has no analytics, tracking scripts, or upload endpoint.
- Models are loaded from the same GitHub Pages site.
- Face IDs are session-local counters. They are not identities and are not stored.

## Accuracy and responsible use

Age, gender, and expression estimation from a webcam is inherently uncertain. Predictions can be affected by lighting, camera angle, occlusion, image quality, and the training data's own limitations and biases.

**Do not use this demo for:**

- identity verification or biometric identification
- surveillance or profiling
- age verification or access control
- medical, employment, housing, insurance, or other high-impact decisions
- determining anyone's gender identity or other personal characteristics

The project is a demonstration of browser-based computer vision, not a validated measurement tool.

## Project layout

```text
index.html                       App shell and UI copy
static/app.css                   Fullscreen camera UI
static/app.js                    Detection, tracking, analysis, and controls
static/models/                   BlazeFace and face-api model files
static/vendor/                   MediaPipe and face-api runtimes
static/vendor/wasm/              MediaPipe WASM binaries
serve.py                         Local static server with correct MIME types
.github/workflows/deploy-pages.yml  GitHub Pages deployment
THIRD_PARTY_NOTICES.md           Third-party licenses and credits
```

## Third-party software

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for licenses and attribution for MediaPipe, face-api.js, TensorFlow.js, and the bundled model weights.
