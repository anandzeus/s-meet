# S-meet - AI-Enhanced Video Call Web App

S-meet is a full-stack, real-time video calling demo with AI-assisted ASL recognition, live text-to-speech, and location sharing. It uses WebRTC for peer-to-peer media and a lightweight Node.js WebSocket server for signaling.

## Features
- 1:1 WebRTC video calling with WebSocket signaling
- ASL recognition with hybrid word + finger-spelling models
- Text-to-speech for received messages (multilingual via browser voices)
- Manual text-to-voice messaging
- Live location sharing (single ping or live tracking)
- Auto call timer

## Architecture
- Client: React + Vite
  - WebRTC for media
  - MediaPipe Holistic for hand + pose landmarks
  - TensorFlow.js for optional client-side ASL inference
  - Browser SpeechSynthesis API for TTS
  - Geolocation API for live location
- Server: Node.js + Express + ws
  - WebSocket signaling for offers, answers, and ICE candidates
  - Room-based message forwarding
- ASL server (optional): FastAPI + PyTorch
  - I3D + Transformer word-level recognition (WLASL)
  - Clip-based inference endpoint for client uploads

## Project structure
```
S-meet/
  asl_server/
    app.py
    model.py
    pytorch_i3d.py
    requirements.txt
    wlasl_class_list.txt
  client/
    src/
      App.jsx
      main.jsx
      styles.css
      hooks/
        useSpeechSynthesis.js
  server/
    server.js
  README.md
```

## Prerequisites
- Node.js 18+ (recommended)
- npm 9+ (comes with Node)
- Python 3.10+ (required for server-side ASL mode)
- A modern browser (Chrome or Edge recommended)
- Camera and microphone permissions enabled

## Local development

### 1) Start the ASL server (optional, for server-side WLASL mode)
```
cd asl_server
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

$env:ASL_MODEL_WEIGHTS="D:\public\S-meet\asl_server\weights\wlasl_pretrained\archived\asl2000\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt"
$env:ASL_LABELS="D:\public\S-meet\asl_server\wlasl_class_list.txt"
$env:ASL_NUM_CLASSES=2000
$env:ASL_MODEL_TYPE=i3d

uvicorn app:app --host 0.0.0.0 --port 8000
```
The ASL server listens on `http://localhost:8000`.

### 2) Start the signaling server
```
cd server
npm install
npm run dev
```
The server listens on `http://localhost:3001` by default.

### 3) Start the client
```
cd client
npm install
npm run dev
```
The client runs on `http://localhost:5173` and connects to the signaling server.

## Configuration

### Client environment variables
Create `client/.env` to override the signaling server URL:
```
VITE_SIGNALING_URL=ws://localhost:3001
VITE_ASL_MODE=hybrid
VITE_ASL_SERVER_URL=http://localhost:8000
VITE_ASL_SERVER_CLIP_MS=1800
VITE_ASL_SERVER_INTERVAL_MS=2400
VITE_ASL_SERVER_CONFIDENCE=0.55
VITE_ASL_RECORDER_MIME=video/webm;codecs=vp8
VITE_ASL_ALPHA_MODEL_TYPE=image
```

### Server environment variables
You can change the server port with:
```
PORT=3001
```

## ASL options

### Server mode (WLASL video model, recommended)
Server mode uses a PyTorch I3D model (word-level WLASL). Download the pretrained weights from:
`https://github.com/dxli94/WLASL` (WLASL pre-trained weights).

Required environment variables:
```
ASL_MODEL_WEIGHTS=D:\public\S-meet\asl_server\weights\wlasl_pretrained\archived\asl2000\FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt
ASL_LABELS=D:\public\S-meet\asl_server\wlasl_class_list.txt
ASL_NUM_CLASSES=2000
ASL_MODEL_TYPE=i3d
```

Notes:
- WLASL is word-level. The UI builds sentences by chaining predictions.
- For spelling or out-of-vocab terms, type in the sentence box manually.
- I3D is the public WLASL2000 weight that ships in the official release.
- `ASL_I3D_WEIGHTS` is only needed when using the I3D+Transformer variant.

### Hybrid mode (server words + alphabet letters)
Hybrid mode uses the server word model plus a client-side alphabet model for finger spelling.
- Word predictions come from the WLASL server.
- Letters come from a TFJS alphabet model (`client/public/models/asl/alpha`).

### Client mode (TFJS landmark models)
Client mode expects two TFJS models:
- Word/phrase model (for whole words)
- Alphabet model (for finger-spelling)

Models are not bundled. Drop in your trained models or hosted URLs.

Place the files in the client public folder:
```
client/public/models/asl/word/model.json
client/public/models/asl/word/labels.json
client/public/models/asl/alpha/model.json
client/public/models/asl/alpha/labels.json
```

You can also point to hosted models with environment variables:
```
VITE_ASL_WORD_MODEL_URL=/models/asl/word/model.json
VITE_ASL_WORD_LABELS_URL=/models/asl/word/labels.json
VITE_ASL_ALPHA_MODEL_URL=/models/asl/alpha/model.json
VITE_ASL_ALPHA_LABELS_URL=/models/asl/alpha/labels.json
VITE_ASL_FEATURE_SET=hands
VITE_ASL_SEQUENCE_LENGTH=32
```

Feature sets:
- `hands` (126 features)
- `hands+pose` (225 features)
- `hands+pose+face` (1629 features)
If the model input size does not match a known set, `VITE_ASL_FEATURE_SET` is used as fallback.

### Alphabet model details
`client/public/models/asl/alpha` includes a starter Sign Language MNIST model (static letters).
It normalizes 28x28 grayscale crops and is best for single-hand finger spelling.
The dataset does not include motion letters (J/Z), so those may be inaccurate.

## Usage guide

### 1) Join a room
- Open the client in two browser tabs or two different devices.
- Use the lobby page to Create room on the first client.
- Share the Room ID or Join link with the other user and click Join room.
- Once connected, the main call workspace appears.

### 2) Start a call
- Click Preview camera to initialize your local stream.
- Click Start call to send the WebRTC offer (host starts the call).

### 3) ASL recognition (hybrid)
- Choose Hybrid mode for WLASL words + alphabet letters.
- Choose Server mode for WLASL words only.
- Choose Client mode for TFJS landmark models (if you have them).
- Click Start capture to enable camera and overlays.
- Server/Hybrid mode: click Capture sign or enable Auto capture to stream short clips.
- Hybrid/Client mode: finger-spelling fills the buffer.
- Click Send sentence to deliver the composed message.
- Use Auto send if you want sentences sent after a short pause.

### 4) Text and voice
- Type a message and click Send to transmit chat.
- Toggle Auto speak to read incoming messages aloud.
- Choose a voice from the dropdown if your browser provides options.

### 5) Live location sharing
- Click Share once to send a one-time location ping.
- Click Start live to continuously share updates until stopped.

### 6) Auto call timer
- Set the duration in minutes and click Start timer.
- The call ends automatically when the timer expires.

## Deployment notes

### Client build
```
cd client
npm run build
```
Host `client/dist` with your static hosting provider.

### Signaling server
- Run the server on a public host (VM, container, or PaaS).
- Point `VITE_SIGNALING_URL` to your server using `wss://` in production.
- If using a reverse proxy, ensure WebSocket upgrades are enabled.

### WebRTC networking
- The demo uses a public Google STUN server.
- For restrictive networks or production usage, configure a TURN server.
- Update `ICE_SERVERS` in `client/src/App.jsx` with your STUN/TURN credentials.

## Security and privacy notes
- This is a demo project with no authentication or access control.
- Media streams are peer-to-peer; the signaling server only relays metadata.
- Location data is sent over the signaling channel to the peer.
- Do not use this as-is for sensitive or production workloads.

## Troubleshooting
- Camera or mic blocked: check browser permission prompts and OS privacy settings.
- Camera unavailable: open the app on `http://localhost:5173` or use HTTPS.
- Signaling not connecting: verify `VITE_SIGNALING_URL` and that port 3001 is reachable.
- Remote video stays black: both peers must be in the room; check NAT restrictions.
- No voices listed: some browsers load voices lazily; reload the page.
- ASL server offline: confirm `uvicorn` is running and `VITE_ASL_SERVER_URL` is correct.
- ASL output missing: verify the models and labels are loaded, and the camera is active.
- Clip decode failed: try `VITE_ASL_RECORDER_MIME=video/webm;codecs=vp8` or Chrome.

## Extending the project
- Add authentication and secure room tokens.
- Integrate a managed TTS service (Google, AWS, etc.) for higher-quality voices.
- Persist chat history and session metadata.
- Add group calls by expanding the signaling server and UI layout.

## License
This project is provided as-is for educational purposes. Add a license if you plan to distribute it.
