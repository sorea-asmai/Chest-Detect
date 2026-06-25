# Bone Fracture Detector

An AI-powered serverless web application designed to accept digital X-ray image uploads and evaluate them for potential bone fractures using an optimized AWS cloud native inference pipeline.

## 🔗 Project Infrastructure & Resources
* **Production Frontend Codebase:** [GitHub Repository Source (`/public`)](https://github.com/Mostgotnom/bone-fracture-detector/tree/main/public)
* **Project Presentation Materials:** [Google Slides Presentation Deck](https://docs.google.com/presentation/d/1u_If1D1Yktz79wrplUABQ6KKKffhqETQE0vV8zWG1HA/edit?usp=sharing)

---

## 🏗️ Detailed Architecture & System Design

The system implements a decoupled single-page frontend combined with a serverless event-driven microservices backend. This avoids the limitations of heavy multipart/form-data multi-boundary streaming natively blocked by default AWS API Gateway route structures.

### 🔁 Deep Data Flow Engineering Pipeline
1. **Asynchronous File Capture:** The user selects a target X-ray image file (`.png` or `.jpeg`). React captures the binary content inside a local DOM upload reference state.
2. **Base64 Marshalling Engine:** Rather than attempting to append raw binary streams to a standard form block, the frontend triggers an asynchronous HTML5 `FileReader` stream reader. This serializes the raw bytes into a flat, base64-encoded UTF-8 ASCII string representation.
3. **Gateway Proxy Handshake:** The frontend dispatches a standard `application/json` payload structure containing the flattened image data. AWS API Gateway matches the preflight `OPTIONS` handshake criteria, injects valid wildcard Access Control headers, and translates the JSON packet safely into a native AWS Lambda event parameter proxy.
4. **Backend Processing & Decoding Array:** The serverless Python runtime parses the proxy body string wrapper. It immediately calls `base64.b64decode()` to restore the structural image bytes directly in RAM memory. This array can then be smoothly ingested by custom deep learning inference models (e.g., PyTorch, TensorFlow, OpenCV, or Pillow).
5. **CORS Structuring Return:** The runtime generates a secure dictionary response object wrapped inside explicit `Access-Control-Allow-Origin: "*"` parameters, preventing downstream Webkit fetch exception drops.

---

## 🛠️ Complete Technical Stack Matrix

### Frontend Architecture
* **Core Library:** React.js (Component-driven SPA structure)
* **Network Pipeline:** Native Window Fetch API (`window.fetch`)
* **State Management Architecture:** React Hooks (`useState` lifecycle trackers)
* **Media Compression/Encoding:** Native Web Browsing HTML5 File API (`FileReader` asynchronous pipeline)
* **Styling Framework:** Custom high-density UI layout elements (`App.css`)

### Backend Serverless Cloud Architecture
* **Inference Compute Engine:** AWS Lambda (Scalable Serverless Compute Node)
* **Traffic Ingress Gatekeeper:** AWS API Gateway (REST Framework Service Integration)
* **Execution Stack Runtime:** Python 3.x
* **Native Module Extensions:** `json`, `base64`, `sys`
