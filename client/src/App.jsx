import React, { useEffect, useMemo, useRef, useState } from 'react';
// MediaPipe is loaded via CDN/script tags in index.html for production stability.
// We keep these as fallbacks or references if the bundler resolves them correctly.
import * as mpHolistic from '@mediapipe/holistic';
import * as mpDrawing from '@mediapipe/drawing_utils';

const Holistic = window.Holistic || mpHolistic.Holistic;
const HAND_CONNECTIONS = window.HAND_CONNECTIONS || mpHolistic.HAND_CONNECTIONS;
const drawConnectors = window.drawConnectors || mpDrawing.drawConnectors;
const drawLandmarks = window.drawLandmarks || mpDrawing.drawLandmarks;
import * as tf from '@tensorflow/tfjs';
import { useSpeechSynthesis } from './hooks/useSpeechSynthesis.js';
import { useSpeechRecognition } from './hooks/useSpeechRecognition.js';

const getDefaultSignalingUrl = () => {
  if (typeof window === 'undefined') {
    return 'ws://127.0.0.1:3001';
  }
  const hn = window.location.hostname;
  const isLocal = hn === 'localhost' || hn === '127.0.0.1' || hn.startsWith('192.168.');
  
  if (isLocal) {
    return `ws://${hn}:3001`;
  }
  
  // In production, assume signaling server is hosted on the same domain or a subdomain
  // If no env var is provided, default to secure wss on the same host
  return `wss://${hn}`;
};

const SIGNALING_URL =
  import.meta.env.VITE_SIGNALING_URL || getDefaultSignalingUrl();

const ICE_SERVERS = [{ urls: 'stun:stun.l.google.com:19302' }];

const DEFAULT_WORD_MODEL_URL = '/models/asl/word/model.json';
const DEFAULT_WORD_LABELS_URL = '/models/asl/word/labels.json';
const DEFAULT_ALPHA_MODEL_URL = '/models/asl/alpha/model.json';
const DEFAULT_ALPHA_LABELS_URL = '/models/asl/alpha/labels.json';

const WORD_MODEL_URL =
  import.meta.env.VITE_ASL_WORD_MODEL_URL || DEFAULT_WORD_MODEL_URL;
const WORD_LABELS_URL =
  import.meta.env.VITE_ASL_WORD_LABELS_URL || DEFAULT_WORD_LABELS_URL;
const ALPHA_MODEL_URL =
  import.meta.env.VITE_ASL_ALPHA_MODEL_URL || DEFAULT_ALPHA_MODEL_URL;
const ALPHA_LABELS_URL =
  import.meta.env.VITE_ASL_ALPHA_LABELS_URL || DEFAULT_ALPHA_LABELS_URL;

const ASL_SEQUENCE_LENGTH = Number(import.meta.env.VITE_ASL_SEQUENCE_LENGTH) || 32;
const ASL_INFERENCE_INTERVAL_MS = 200;
const ASL_WORD_STABLE_MS = 500;
const ASL_WORD_RELEASE_MS = 600;
const ASL_WORD_COOLDOWN_MS = 1400;
const ASL_WORD_CONFIDENCE = 0.75;
const ASL_ALPHA_STABLE_MS = 200;
const ASL_ALPHA_RELEASE_MS = 400;
const ASL_ALPHA_COOLDOWN_MS = 350;
const ASL_ALPHA_CONFIDENCE = 0.65;
const ASL_ALPHA_MODEL_TYPE = (
  import.meta.env.VITE_ASL_ALPHA_MODEL_TYPE || 'server'
).toLowerCase();
const ASL_ALPHA_IMAGE_INTERVAL_MS = 100;
const ASL_ALPHA_SERVER_INTERVAL_MS = 100;
const ASL_LETTER_PAUSE_MS = 1100;
const ASL_SENTENCE_PAUSE_MS = 2600;
const getDefaultAslUrl = () => {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8000';
  const hn = window.location.hostname;
  const isLocal = hn === 'localhost' || hn === '127.0.0.1' || hn.startsWith('192.168.');
  // If local development, use port 8000. If production, rely on env var or assume same host
  return isLocal ? `http://${hn}:8000` : `https://${hn}`;
};

const ASL_SERVER_URL = import.meta.env.VITE_ASL_SERVER_URL || getDefaultAslUrl();
const ASL_SERVER_CLIP_MS =
  Number(import.meta.env.VITE_ASL_SERVER_CLIP_MS) || 1800;
const ASL_SERVER_INTERVAL_MS =
  Number(import.meta.env.VITE_ASL_SERVER_INTERVAL_MS) || 2400;
const ASL_SERVER_CONFIDENCE =
  Number(import.meta.env.VITE_ASL_SERVER_CONFIDENCE) || 0.55;
const ASL_RECORDER_MIME = import.meta.env.VITE_ASL_RECORDER_MIME || '';
const DEFAULT_ASL_MODE = import.meta.env.VITE_ASL_MODE || 'server';

const FEATURE_SETS = {
  hands: { key: 'hands', includePose: false, includeFace: false, size: 126 },
  handsPose: { key: 'hands+pose', includePose: true, includeFace: false, size: 225 },
  handsPoseFace: { key: 'hands+pose+face', includePose: true, includeFace: true, size: 1629 }
};

const PREFERRED_FEATURE_SET = import.meta.env.VITE_ASL_FEATURE_SET || '';

const ROOM_WORDS = [
  'aurora',
  'cinder',
  'delta',
  'ember',
  'harbor',
  'lumen',
  'orbit',
  'pulse',
  'rally',
  'river',
  'sierra',
  'solace',
  'spark',
  'tango',
  'vista',
  'zenith'
];

const makeId = () => {
  try {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }
  } catch (e) {
    // Fallback if randomUUID throws due to insecure context on iOS
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const getRandomInt = (max) => {
  try {
    if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
      const values = new Uint32Array(1);
      crypto.getRandomValues(values);
      return values[0] % max;
    }
  } catch (e) {
    // Fallback if crypto throws an error (e.g. self-signed cert on iOS)
  }
  return Math.floor(Math.random() * max);
};

const makeRoomCode = () => {
  const first = ROOM_WORDS[getRandomInt(ROOM_WORDS.length)];
  const second = ROOM_WORDS[getRandomInt(ROOM_WORDS.length)];
  const suffix = getRandomInt(900) + 100;
  return `${first}-${second}-${suffix}`;
};

const buildRoomLink = (room) => {
  if (typeof window === 'undefined') {
    return room;
  }
  return `${window.location.origin}?room=${encodeURIComponent(room)}`;
};

const attachStreamToVideo = (videoEl, stream) => {
  if (!videoEl || !stream) {
    return;
  }
  if (videoEl.srcObject !== stream) {
    videoEl.srcObject = stream;
  }
  const playPromise = videoEl.play?.();
  if (playPromise && typeof playPromise.catch === 'function') {
    playPromise.catch(() => {});
  }
};

const getMediaSupport = () => {
  if (typeof window === 'undefined') {
    return { ok: false, reason: 'unsupported environment' };
  }
  if (!window.isSecureContext) {
    return { ok: false, reason: 'requires https or localhost' };
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return { ok: false, reason: 'camera unavailable (try Chrome or Edge)' };
  }
  return { ok: true, reason: 'ready' };
};

const describeMediaError = (error) => {
  if (!error) {
    return 'camera error';
  }
  if (error.name === 'NotAllowedError') {
    return 'permission denied';
  }
  if (error.name === 'NotFoundError') {
    return 'no camera found';
  }
  if (error.name === 'NotReadableError') {
    return 'camera in use';
  }
  if (error.name === 'OverconstrainedError') {
    return 'device constraints not met';
  }
  return 'camera error';
};

const formatTimer = (ms) => {
  if (ms == null) {
    return '0:00';
  }
  const clamped = Math.max(0, ms);
  const totalSeconds = Math.floor(clamped / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
};

const normalizeGloss = (label) => {
  if (label === null || label === undefined) {
    return '';
  }
  const text = String(label);
  if (!text) {
    return '';
  }
  return text.replace(/_/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
};

const formatPrediction = (label, score) => {
  if (!label) {
    return '...';
  }
  if (!score) {
    return label;
  }
  return `${label} ${Math.round(score * 100)}%`;
};

const getHandArea = (landmarks) => {
  if (!landmarks || !landmarks.length) {
    return 0;
  }
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  landmarks.forEach((point) => {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });
  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  return width * height;
};

const chooseHandLandmarks = (leftHand, rightHand) => {
  if (leftHand && rightHand) {
    const leftArea = getHandArea(leftHand);
    const rightArea = getHandArea(rightHand);
    return leftArea >= rightArea ? leftHand : rightHand;
  }
  return leftHand || rightHand || null;
};

const getHandBox = (landmarks, padding = 0.12) => {
  if (!landmarks || !landmarks.length) {
    return null;
  }
  let minX = 1;
  let minY = 1;
  let maxX = 0;
  let maxY = 0;
  landmarks.forEach((point) => {
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
  });
  minX = Math.max(0, minX - padding);
  minY = Math.max(0, minY - padding);
  maxX = Math.min(1, maxX + padding);
  maxY = Math.min(1, maxY + padding);
  if (maxX <= minX || maxY <= minY) {
    return null;
  }
  return { minX, minY, maxX, maxY };
};

const buildImageSpec = (model) => {
  const shape = model?.inputs?.[0]?.shape || [];
  const height = shape[1] || 28;
  const width = shape[2] || 28;
  const channels = shape[3] || 1;
  return { type: 'image', height, width, channels };
};

const pickRecorderMimeType = () => {
  if (ASL_RECORDER_MIME) {
    return ASL_RECORDER_MIME;
  }
  if (typeof MediaRecorder === 'undefined') {
    return '';
  }
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm'
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
};

const appendLandmarks = (target, landmarks, count) => {
  if (!landmarks || landmarks.length < count) {
    for (let i = 0; i < count; i += 1) {
      target.push(0, 0, 0);
    }
    return;
  }
  for (let i = 0; i < count; i += 1) {
    const point = landmarks[i];
    target.push(point?.x ?? 0, point?.y ?? 0, point?.z ?? 0);
  }
};

const buildFeatureVector = (results, featureSet) => {
  const features = [];
  if (featureSet.includePose) {
    appendLandmarks(features, results.poseLandmarks, 33);
  }
  if (featureSet.includeFace) {
    appendLandmarks(features, results.faceLandmarks, 468);
  }
  appendLandmarks(features, results.leftHandLandmarks, 21);
  appendLandmarks(features, results.rightHandLandmarks, 21);
  return features;
};

const resolveFeatureSet = (featureSize) => {
  const match = Object.values(FEATURE_SETS).find((entry) => entry.size === featureSize);
  if (match) {
    return match;
  }
  const preferred = Object.values(FEATURE_SETS).find(
    (entry) => entry.key === PREFERRED_FEATURE_SET
  );
  return preferred || FEATURE_SETS.hands;
};

const parseLabels = (data) => {
  if (Array.isArray(data)) {
    return data;
  }
  if (data && Array.isArray(data.labels)) {
    return data.labels;
  }
  return [];
};

export default function App() {
  const [roomId, setRoomId] = useState('');
  const [name, setName] = useState('');
  const [status, setStatus] = useState('disconnected');
  const [connected, setConnected] = useState(false);
  const [clientId, setClientId] = useState('');
  const [isInitiator, setIsInitiator] = useState(false);
  const [callActive, setCallActive] = useState(false);
  const [localStream, setLocalStream] = useState(null);
  const [messages, setMessages] = useState([]);
  const [messageInput, setMessageInput] = useState('');
  const [gestureEnabled, setGestureEnabled] = useState(false);
  const [gestureSummary, setGestureSummary] = useState('ASL capture idle');
  const [gestureDraft, setGestureDraft] = useState('');
  const [letterBuffer, setLetterBuffer] = useState('');
  const [aslMode, setAslMode] = useState(DEFAULT_ASL_MODE);
  const [aslAutoCapture, setAslAutoCapture] = useState(true);
  const [aslAutoAppend, setAslAutoAppend] = useState(true);
  const [aslAutoSend, setAslAutoSend] = useState(true);
  const [aslServerStatus, setAslServerStatus] = useState('idle');
  const [aslServerPrediction, setAslServerPrediction] = useState({
    label: '',
    score: 0
  });
  const [aslServerBusy, setAslServerBusy] = useState(false);
  const [aslWordStatus, setAslWordStatus] = useState('not loaded');
  const [aslAlphaStatus, setAslAlphaStatus] = useState('not loaded');
  const [aslWordPrediction, setAslWordPrediction] = useState({
    label: '',
    score: 0
  });
  const [aslAlphaPrediction, setAslAlphaPrediction] = useState({
    label: '',
    score: 0
  });
  const [autoSpeak, setAutoSpeak] = useState(true);
  const [selectedVoice, setSelectedVoice] = useState('');
  const [timerMinutes, setTimerMinutes] = useState('15');
  const [timerActive, setTimerActive] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(null);
  const [remoteLocation, setRemoteLocation] = useState(null);
  const [locationStatus, setLocationStatus] = useState('idle');
  const [locationSharing, setLocationSharing] = useState(false);
  const [micEnabled, setMicEnabled] = useState(true);
  const [camEnabled, setCamEnabled] = useState(true);
  const [remoteStreamVersion, setRemoteStreamVersion] = useState(0);
  const [roomHint, setRoomHint] = useState('');
  const [deviceStatus, setDeviceStatus] = useState('idle');
  
  // STT State
  const [sttEnabled, setSttEnabled] = useState(true);
  const [remoteSttText, setRemoteSttText] = useState('');
  const remoteSttTimerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const wsRef = useRef(null);
  const pcsRef = useRef(new Map());
  const localStreamRef = useRef(null);
  const remoteStreamsRef = useRef(new Map());
  const localVideoRef = useRef(null);
  const fsLocalVideoRef = useRef(null);
  const fsRemoteVideoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const holisticRef = useRef(null);
  const gestureLoopRef = useRef(null);
  const aslWordModelRef = useRef(null);
  const aslAlphaModelRef = useRef(null);
  const aslWordLabelsRef = useRef([]);
  const aslAlphaLabelsRef = useRef([]);
  const aslWordSpecRef = useRef(null);
  const aslAlphaSpecRef = useRef(null);
  const aslWordSequenceRef = useRef([]);
  const aslAlphaSequenceRef = useRef([]);
  const aslWordStateRef = useRef({
    candidate: null,
    candidateSince: 0,
    lastSent: null,
    lastSentAt: 0,
    lastClearAt: 0
  });
  const aslAlphaStateRef = useRef({
    candidate: null,
    candidateSince: 0,
    lastSent: null,
    lastSentAt: 0,
    lastClearAt: 0
  });
  const lastLetterAtRef = useRef(0);
  const lastSentenceActivityRef = useRef(0);
  const gestureEnabledRef = useRef(gestureEnabled);
  const gestureSummaryRef = useRef(gestureSummary);
  const aslModeRef = useRef(aslMode);
  const aslAutoCaptureRef = useRef(aslAutoCapture);
  const aslServerBusyRef = useRef(aslServerBusy);
  const aslServerPredictionRef = useRef(aslServerPrediction);
  const aslAlphaPredictionRef = useRef(aslAlphaPrediction);
  const lastServerPredictionRef = useRef({ label: '', at: 0 });
  const mediaRecorderRef = useRef(null);
  const captureTimeoutRef = useRef(null);
  const serverCaptureIntervalRef = useRef(null);
  const lastAutoSendRef = useRef(0);
  const lastInferenceRef = useRef(0);
  const inferenceInFlightRef = useRef(false);
  const lastAlphaImageInferenceRef = useRef(0);
  const alphaImageInFlightRef = useRef(false);
  const lastAlphaServerInferenceRef = useRef(0);
  const alphaServerInFlightRef = useRef(false);
  const gestureDraftRef = useRef('');
  const letterBufferRef = useRef('');
  const aslAutoAppendRef = useRef(aslAutoAppend);
  const aslAutoSendRef = useRef(aslAutoSend);
  const modelLoadRef = useRef(false);
  const locationWatchRef = useRef(null);
  const timerTimeoutRef = useRef(null);
  const timerIntervalRef = useRef(null);
  const roomHintTimeoutRef = useRef(null);
  const autoSpeakRef = useRef(autoSpeak);
  const selectedVoiceRef = useRef(selectedVoice);
  const voicesRef = useRef([]);
  const nameRef = useRef(name);

  // Initialize Speech Recognition Hook
  const { 
    supported: sttSupported, 
    listening: sttListening, 
    transcript: sttTranscript, 
    startListening: startStt, 
    stopListening: stopStt 
  } = useSpeechRecognition();

  const { voices, speak, supported: ttsSupported } = useSpeechSynthesis();

  const voiceOptions = useMemo(
    () =>
      [...(voices || [])]
        .sort((a, b) => a.lang.localeCompare(b.lang))
        .map((voice) => ({
          label: `${voice.lang} - ${voice.name}`,
          value: voice.voiceURI
        })),
    [voices]
  );

  useEffect(() => {
    autoSpeakRef.current = autoSpeak;
  }, [autoSpeak]);

  useEffect(() => {
    gestureSummaryRef.current = gestureSummary;
  }, [gestureSummary]);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  useEffect(() => {
    aslAutoAppendRef.current = aslAutoAppend;
  }, [aslAutoAppend]);

  useEffect(() => {
    aslAutoSendRef.current = aslAutoSend;
  }, [aslAutoSend]);

  useEffect(() => {
    aslModeRef.current = aslMode;
  }, [aslMode]);

  useEffect(() => {
    aslAutoCaptureRef.current = aslAutoCapture;
  }, [aslAutoCapture]);

  useEffect(() => {
    aslServerBusyRef.current = aslServerBusy;
  }, [aslServerBusy]);

  useEffect(() => {
    gestureEnabledRef.current = gestureEnabled;
  }, [gestureEnabled]);

  useEffect(() => {
    aslServerPredictionRef.current = aslServerPrediction;
  }, [aslServerPrediction]);

  useEffect(() => {
    aslAlphaPredictionRef.current = aslAlphaPrediction;
  }, [aslAlphaPrediction]);

  useEffect(() => {
    voicesRef.current = voices;
  }, [voices]);

  useEffect(() => {
    gestureDraftRef.current = gestureDraft;
  }, [gestureDraft]);

  useEffect(() => {
    letterBufferRef.current = letterBuffer;
  }, [letterBuffer]);

  useEffect(() => {
    nameRef.current = name;
  }, [name]);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }
    const params = new URLSearchParams(window.location.search);
    const roomFromUrl = params.get('room');
    if (roomFromUrl) {
      setRoomId(roomFromUrl);
    }
  }, []);

  useEffect(() => {
    const support = getMediaSupport();
    if (!support.ok) {
      setDeviceStatus(support.reason);
    }
  }, []);

  useEffect(() => {
    if (connected && !gestureEnabled) {
      startGestures();
    }
  }, [connected]);

  // STT Control Effect
  useEffect(() => {
    if (!sttSupported) return;
    // STT should run if explicitly enabled, mic is unmuted, and local stream exists (even before call is active)
    if (localStream && sttEnabled && micEnabled) {
      if (!sttListening) {
        startStt();
      }
    } else {
      if (sttListening) {
        stopStt();
      }
    }
  }, [localStream, callActive, sttEnabled, micEnabled, sttSupported, sttListening, startStt, stopStt]);

  // STT Transcript Sync Effect
  useEffect(() => {
    if (callActive && sttTranscript) {
      sendWs({ type: 'stt', text: sttTranscript });
    }
  }, [sttTranscript, callActive]);

  useEffect(() => {
    if (gestureEnabled && aslAutoCapture && (aslMode === 'server' || aslMode === 'hybrid')) {
      startServerAutoCapture();
    } else if (!aslAutoCapture) {
      stopServerAutoCapture();
    }
  }, [gestureEnabled, aslAutoCapture, aslMode]);

  useEffect(() => {
    if (localVideoRef.current && localStream) {
      attachStreamToVideo(localVideoRef.current, localStream);
    }
  }, [localStream]);

  // Remote videos are attached inline via React refs

  const sendWs = (payload) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    ws.send(JSON.stringify(payload));
  };

  const setRoomHintMessage = (text) => {
    if (roomHintTimeoutRef.current) {
      clearTimeout(roomHintTimeoutRef.current);
      roomHintTimeoutRef.current = null;
    }
    setRoomHint(text);
    if (text) {
      roomHintTimeoutRef.current = setTimeout(() => {
        setRoomHint('');
        roomHintTimeoutRef.current = null;
      }, 4000);
    }
  };

  const pushMessage = (entry) => {
    setMessages((prev) => [...prev, entry].slice(-100));
  };

  const pushSystemMessage = (text) => {
    pushMessage({
      id: makeId(),
      type: 'system',
      text,
      from: 'System',
      self: false,
      timestamp: new Date().toISOString()
    });
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  // Sync streams to video elements, especially for fullscreen stabilization
  useEffect(() => {
    if (isFullscreen) {
      if (fsRemoteVideoRef.current) {
        const firstStream = Array.from(remoteStreamsRef.current.values())[0] || new MediaStream();
        fsRemoteVideoRef.current.srcObject = firstStream;
      }
      if (fsLocalVideoRef.current && localStreamRef.current) {
        fsLocalVideoRef.current.srcObject = localStreamRef.current;
      }
    }
  }, [isFullscreen, remoteStreamVersion, localStream]);

  const ensureLocalStream = async () => {
    if (localStreamRef.current) {
      setDeviceStatus('ready');
      return localStreamRef.current;
    }
    const support = getMediaSupport();
    if (!support.ok) {
      setDeviceStatus(support.reason);
      return null;
    }
    setDeviceStatus('requesting access');
    setGestureSummary('Requesting camera access...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      localStreamRef.current = stream;
      setLocalStream(stream);
      setMicEnabled(stream.getAudioTracks().some((track) => track.enabled));
      setCamEnabled(stream.getVideoTracks().some((track) => track.enabled));
      setDeviceStatus('ready');
      return stream;
    } catch (error) {
      const errDesc = describeMediaError(error);
      setDeviceStatus(errDesc);
      setGestureSummary(`Camera error: ${errDesc}`);
      return null;
    }
  };

  const stopLocalStream = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => track.stop());
      localStreamRef.current = null;
    }
    setLocalStream(null);
    setMicEnabled(false);
    setCamEnabled(false);
    const support = getMediaSupport();
    setDeviceStatus(support.ok ? 'idle' : support.reason);
  };

  const resetRemoteStreams = () => {
    pcsRef.current.forEach(pc => pc.close());
    pcsRef.current.clear();
    remoteStreamsRef.current.clear();
    setRemoteStreamVersion((version) => version + 1);
  };

  const createPeerConnection = (targetId) => {
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS });

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        sendWs({ type: 'signal', target: targetId, candidate: event.candidate });
      }
    };

    pc.ontrack = (event) => {
      let stream = remoteStreamsRef.current.get(targetId);
      if (!stream) {
        stream = new MediaStream();
        remoteStreamsRef.current.set(targetId, stream);
      }
      event.streams[0].getTracks().forEach((track) => {
        if (!stream.getTracks().includes(track)) {
          stream.addTrack(track);
        }
      });
      setRemoteStreamVersion((version) => version + 1);
    };

    pc.onconnectionstatechange = () => {
      let active = false;
      pcsRef.current.forEach(p => { if (p.connectionState === 'connected') active = true; });
      setCallActive(active);

      if (
        pc.connectionState === 'failed' ||
        pc.connectionState === 'disconnected' ||
        pc.connectionState === 'closed'
      ) {
        pcsRef.current.delete(targetId);
        remoteStreamsRef.current.delete(targetId);
        setRemoteStreamVersion((version) => version + 1);
      }
    };

    pcsRef.current.set(targetId, pc);
    return pc;
  };

  const handleSignal = async (message) => {
    const fromId = message.from?.id;
    if (!fromId) return;

    let pc = pcsRef.current.get(fromId);
    if (!pc) {
      pc = createPeerConnection(fromId);
    }

    if (message.sdp) {
      await pc.setRemoteDescription(new RTCSessionDescription(message.sdp));

      if (message.sdp.type === 'offer') {
        const stream = await ensureLocalStream();
        if (stream) {
          const senders = pc.getSenders();
          stream.getTracks().forEach((track) => {
            if (!senders.find((s) => s.track === track)) {
              pc.addTrack(track, stream);
            }
          });
        }
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        sendWs({ type: 'signal', target: fromId, sdp: pc.localDescription });
      }
    }

    if (message.candidate) {
      try {
        await pc.addIceCandidate(new RTCIceCandidate(message.candidate));
      } catch (error) {
        setStatus('ICE candidate error');
      }
    }
  };

  const handleWsMessage = async (message) => {
    if (message.type === 'room-info') {
      setClientId(message.id || '');
      setIsInitiator(message.peers === 1);
      setStatus(message.peers > 1 ? 'ready to call' : 'waiting for peer');
      return;
    }

    if (message.type === 'peer-joined') {
      pushSystemMessage(`${message.name || 'Peer'} joined.`);
      setStatus('peer joined');
      if (callActive || localStreamRef.current) {
        setTimeout(async () => {
          const pc = createPeerConnection(message.id);
          const stream = await ensureLocalStream();
          if (stream) {
            const senders = pc.getSenders();
            stream.getTracks().forEach((track) => {
              if (!senders.find((s) => s.track === track)) pc.addTrack(track, stream);
            });
          }
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          sendWs({ type: 'signal', target: message.id, sdp: pc.localDescription });
        }, 500); 
      }
      return;
    }

    if (message.type === 'peer-left') {
      pushSystemMessage(`${message.name || 'Peer'} left.`);
      setStatus('peer left');
      const pc = pcsRef.current.get(message.id);
      if (pc) pc.close();
      pcsRef.current.delete(message.id);
      remoteStreamsRef.current.delete(message.id);
      setRemoteStreamVersion((v) => v + 1);
      
      if (pcsRef.current.size === 0) {
        setCallActive(false);
      }
      return;
    }

    if (message.type === 'call-start') {
      const targetId = message.from?.id;
      if (targetId && !pcsRef.current.has(targetId)) {
        setTimeout(async () => {
          const pc = createPeerConnection(targetId);
          const stream = await ensureLocalStream();
          if (stream) {
            const senders = pc.getSenders();
            stream.getTracks().forEach((track) => {
              if (!senders.find((s) => s.track === track)) pc.addTrack(track, stream);
            });
          }
          const offer = await pc.createOffer();
          await pc.setLocalDescription(offer);
          sendWs({ type: 'signal', target: targetId, sdp: pc.localDescription });
        }, 500);
      }
      return;
    }

    if (message.type === 'signal') {
      await handleSignal(message);
      return;
    }

    if (message.type === 'chat' || message.type === 'gesture') {
      const text = message.text || '';
      const fromName = message.from?.name || 'Peer';
      const entry = {
        id: makeId(),
        type: message.type,
        text,
        from: fromName,
        self: false,
        timestamp: new Date().toISOString()
      };
      pushMessage(entry);

      if (autoSpeakRef.current && ttsSupported) {
        const voice = voicesRef.current.find(
          (item) => item.voiceURI === selectedVoiceRef.current
        );
        speak(text, voice);
      }
      return;
    }

    if (message.type === 'location') {
      const coords = message.coords || {};
      setRemoteLocation({
        lat: coords.lat,
        lng: coords.lng,
        accuracy: coords.accuracy,
        timestamp: message.timestamp,
        from: message.from?.name || 'Peer'
      });
      return;
    }

    if (message.type === 'stt') {
      setRemoteSttText(message.text || '');
      if (remoteSttTimerRef.current) clearTimeout(remoteSttTimerRef.current);
      const timer = setTimeout(() => {
        setRemoteSttText('');
      }, 3000);
      remoteSttTimerRef.current = timer;
      return;
    }

    if (message.type === 'hangup') {
      endCall(false);
    }
  };

  const connectSocket = (roomOverride) => {
    const trimmedRoom = String(roomOverride ?? roomId).trim();
    if (!trimmedRoom) {
      setStatus('room id required');
      setRoomHintMessage('Enter a room id to continue.');
      return;
    }

    if (wsRef.current) {
      wsRef.current.close();
    }

    setRoomId(trimmedRoom);
    setStatus('connecting');
    const ws = new WebSocket(SIGNALING_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setStatus('connected');
      ws.send(
        JSON.stringify({
          type: 'join',
          roomId: trimmedRoom,
          name: name.trim() || 'Guest'
        })
      );
    };

    ws.onmessage = async (event) => {
      let parsed;
      try {
        parsed = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      await handleWsMessage(parsed);
    };

    ws.onclose = () => {
      setConnected(false);
      setStatus('disconnected');
      setIsInitiator(false);
      setClientId('');
    };
  };

  const disconnectSocket = () => {
    endCall(true);
    stopGestures();
    stopLocalStream();
    setGestureDraft('');
    sendWs({ type: 'leave' });
    if (wsRef.current) {
      wsRef.current.close();
    }
    wsRef.current = null;
    setConnected(false);
    setStatus('disconnected');
    setIsInitiator(false);
    setClientId('');
    setRoomHintMessage('Left the room.');
  };

  const createRoom = () => {
    const newRoom = makeRoomCode();
    connectSocket(newRoom);
    setRoomHintMessage('Room created and joined.');
  };

  const copyToClipboard = async (text, label) => {
    if (!text) {
      setRoomHintMessage('Enter a room id first.');
      return;
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        setRoomHintMessage(`${label} copied.`);
      } else {
        setRoomHintMessage('Clipboard unavailable in this browser.');
      }
    } catch (error) {
      setRoomHintMessage('Copy failed.');
    }
  };

  const copyRoomId = () => copyToClipboard(roomId.trim(), 'Room id');
  const copyRoomLink = () =>
    copyToClipboard(buildRoomLink(roomId.trim()), 'Join link');

  const startCall = async () => {
    if (!connected) {
      setStatus('connect first');
      return;
    }

    const stream = await ensureLocalStream();
    if (!stream) {
      return;
    }
    
    setCallActive(true);
    sendWs({ type: 'call-start' });
  };

  const clearTimer = () => {
    if (timerTimeoutRef.current) {
      clearTimeout(timerTimeoutRef.current);
      timerTimeoutRef.current = null;
    }
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }
    setTimerActive(false);
    setTimeRemaining(null);
  };

  const endCall = (notify = true) => {
    if (notify) sendWs({ type: 'hangup' });
    
    pcsRef.current.forEach(pc => {
      pc.ontrack = null;
      pc.onicecandidate = null;
      pc.close();
    });
    setCallActive(false);
    resetRemoteStreams();
    clearTimer();
  };

  const toggleTrack = (kind) => {
    const stream = localStreamRef.current;
    if (!stream) {
      return;
    }

    stream.getTracks().forEach((track) => {
      if (track.kind === kind) {
        track.enabled = !track.enabled;
      }
    });

    setMicEnabled(stream.getAudioTracks().some((track) => track.enabled));
    setCamEnabled(stream.getVideoTracks().some((track) => track.enabled));
  };

  const sendChat = () => {
    const text = messageInput.trim();
    if (!text || !connected) {
      return;
    }

    pushMessage({
      id: makeId(),
      type: 'chat',
      text,
      from: nameRef.current || 'You',
      self: true,
      timestamp: new Date().toISOString()
    });

    sendWs({ type: 'chat', text });
    setMessageInput('');
  };

  const appendSentenceWord = (word) => {
    if (!word) {
      return;
    }
    setGestureDraft((prev) => (prev ? `${prev} ${word}` : word));
    lastSentenceActivityRef.current = Date.now();
  };

  const appendLetter = (letter) => {
    if (!letter) {
      return;
    }
    const next = `${letterBufferRef.current}${letter}`;
    letterBufferRef.current = next;
    setLetterBuffer(next);
    lastLetterAtRef.current = Date.now();
    lastSentenceActivityRef.current = Date.now();
  };

  const deleteLastLetter = () => {
    const next = letterBufferRef.current.slice(0, -1);
    letterBufferRef.current = next;
    setLetterBuffer(next);
    lastLetterAtRef.current = Date.now();
    lastSentenceActivityRef.current = Date.now();
  };

  const normalizeHandLabel = (label) => {
    if (!label) return '';
    if (label.length === 1) return label.toUpperCase();
    return label
      .replace(/_/g, ' ')
      .replace(/thankyou/i, 'thank you')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const commitLetterBuffer = () => {
    const buffer = letterBufferRef.current.trim();
    if (!buffer) {
      return;
    }
    letterBufferRef.current = '';
    setLetterBuffer('');
    appendSentenceWord(buffer);
  };

  const clearGestureDraft = () => {
    setGestureDraft('');
    gestureDraftRef.current = '';
    setLetterBuffer('');
    letterBufferRef.current = '';
  };

  const sendGestureText = (text, source) => {
    if (!text || !connected) {
      return;
    }
    pushMessage({
      id: makeId(),
      type: 'gesture',
      text,
      from: nameRef.current || 'You',
      self: true,
      timestamp: new Date().toISOString()
    });
    sendWs({ type: 'gesture', text, source });
  };

  const sendGestureDraft = () => {
    const text = gestureDraftRef.current.trim();
    const buffer = letterBufferRef.current.trim();
    const combined = [text, buffer].filter(Boolean).join(' ');
    if (!combined || !connected) {
      return;
    }
    sendGestureText(combined, 'manual');
    setGestureDraft('');
    gestureDraftRef.current = '';
    setLetterBuffer('');
    letterBufferRef.current = '';
  };

  const probeAslServer = async () => {
    if (!ASL_SERVER_URL) {
      setAslServerStatus('missing url');
      return;
    }
    setAslServerStatus('checking');
    try {
      const response = await fetch(`${ASL_SERVER_URL}/health`, {
        headers: {
        }
      });
      if (!response.ok) {
        throw new Error('health check failed');
      }
      const data = await response.json();
      setAslServerStatus(data.ok && data.alpha_model_loaded ? 'ready' : 'error');
      setAslAlphaStatus(data.alpha_model_loaded ? 'loaded' : 'missing');
    } catch (error) {
      setAslServerStatus('offline');
    }
  };

  const applyServerPrediction = (label, score) => {
    const normalized = normalizeGloss(label);
    const display = normalized || label || '';
    setAslServerPrediction({ label: display, score: score || 0 });
    const serverText = formatPrediction(display, score || 0);
    if (aslModeRef.current === 'hybrid') {
      const alpha = aslAlphaPredictionRef.current;
      const alphaText = formatPrediction(alpha?.label, alpha?.score);
      setGestureSummary(`Server: ${serverText} | Alpha: ${alphaText}`);
    } else {
      setGestureSummary(`Server: ${serverText}`);
    }

    if (!display || (score || 0) < ASL_SERVER_CONFIDENCE) {
      return;
    }

    const now = Date.now();
    const last = lastServerPredictionRef.current;
    if (now - last.at < ASL_WORD_COOLDOWN_MS && last.label === display) {
      return;
    }
    lastServerPredictionRef.current = { label: display, at: now };
    if (aslAutoAppendRef.current) {
      appendSentenceWord(display);
    }
  };

  const sendClipForPrediction = async (blob) => {
    if (!ASL_SERVER_URL) {
      setAslServerStatus('missing url');
      setGestureSummary('ASL server url missing');
      return;
    }
    aslServerBusyRef.current = true;
    setAslServerBusy(true);
    try {
      const form = new FormData();
      form.append('file', blob, 'clip.webm');
      const response = await fetch(`${ASL_SERVER_URL}/asl/predict`, {
        method: 'POST',
        headers: {
        },
        body: form
      });
      if (!response.ok) {
        throw new Error('prediction failed');
      }
      const data = await response.json();
      applyServerPrediction(data.label, data.score);
      setAslServerStatus('ready');
    } catch (error) {
      setAslServerStatus('offline');
      setGestureSummary('ASL server offline');
    } finally {
      aslServerBusyRef.current = false;
      setAslServerBusy(false);
    }
  };

  const stopServerRecording = () => {
    if (captureTimeoutRef.current) {
      clearTimeout(captureTimeoutRef.current);
      captureTimeoutRef.current = null;
    }
    if (mediaRecorderRef.current) {
      const recorder = mediaRecorderRef.current;
      if (recorder.state !== 'inactive') {
        recorder.stop();
      }
      mediaRecorderRef.current = null;
    }
    aslServerBusyRef.current = false;
    setAslServerBusy(false);
  };

  const captureServerClip = async () => {
    if (aslServerBusyRef.current) {
      return;
    }
    const stream = await ensureLocalStream();
    if (!stream) {
      return;
    }
    const videoTracks = stream.getVideoTracks();
    if (!videoTracks.length) {
      setGestureSummary('Camera disabled');
      return;
    }

    const mimeType = pickRecorderMimeType();
    if (!mimeType) {
      setGestureSummary('MediaRecorder not supported');
      return;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      return;
    }

    let recorder;
    const recordStream = new MediaStream(videoTracks);
    try {
      recorder = new MediaRecorder(recordStream, { mimeType });
    } catch (error) {
      aslServerBusyRef.current = false;
      setAslServerBusy(false);
      setGestureSummary('Recording not supported');
      return;
    }
    mediaRecorderRef.current = recorder;
    const chunks = [];
    aslServerBusyRef.current = true;
    setAslServerBusy(true);
    setGestureSummary('Recording clip...');

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunks.push(event.data);
      }
    };

    recorder.onerror = () => {
      aslServerBusyRef.current = false;
      setAslServerBusy(false);
      setGestureSummary('Recording failed');
    };

    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      sendClipForPrediction(blob);
    };

    recorder.start();
    captureTimeoutRef.current = setTimeout(() => {
      if (recorder.state === 'recording') {
        recorder.stop();
      }
    }, ASL_SERVER_CLIP_MS);
  };

  const startServerAutoCapture = () => {
    if (serverCaptureIntervalRef.current) {
      return;
    }
    const interval = Math.max(ASL_SERVER_INTERVAL_MS, ASL_SERVER_CLIP_MS + 200);
    serverCaptureIntervalRef.current = setInterval(() => {
      if (
        !gestureEnabledRef.current ||
        (aslModeRef.current !== 'server' && aslModeRef.current !== 'hybrid')
      ) {
        return;
      }
      captureServerClip();
    }, interval);
  };

  const stopServerAutoCapture = () => {
    if (serverCaptureIntervalRef.current) {
      clearInterval(serverCaptureIntervalRef.current);
      serverCaptureIntervalRef.current = null;
    }
  };

  const toggleAutoCapture = () => {
    setAslAutoCapture((prev) => {
      const next = !prev;
      if (
        next &&
        gestureEnabled &&
        (aslModeRef.current === 'server' || aslModeRef.current === 'hybrid')
      ) {
        startServerAutoCapture();
      } else if (!next) {
        stopServerAutoCapture();
      }
      return next;
    });
  };

  const handleAslModeChange = (event) => {
    const nextMode = event.target.value;
    setAslMode(nextMode);
    if (nextMode !== 'server' && nextMode !== 'hybrid') {
      stopServerAutoCapture();
      setAslAutoCapture(false);
    }
  };

  const getTopPrediction = (scores, labels) => {
    if (!scores || scores.length === 0) {
      return { label: '', score: 0 };
    }
    let bestIndex = 0;
    for (let i = 1; i < scores.length; i += 1) {
      if (scores[i] > scores[bestIndex]) {
        bestIndex = i;
      }
    }
    const label = labels[bestIndex] || `class-${bestIndex}`;
    return { label, score: scores[bestIndex] };
  };

  const updatePredictionState = (stateRef, label, score, config, now) => {
    const state = stateRef.current;
    if (!label || score < config.minScore) {
      state.candidate = null;
      state.lastClearAt = now;
      return false;
    }

    if (state.candidate !== label) {
      state.candidate = label;
      state.candidateSince = now;
    }

    const stable = now - state.candidateSince >= config.stableMs;
    const cooldownOk = now - state.lastSentAt >= config.cooldownMs;
    const releaseOk =
      state.lastSent !== label || now - state.lastClearAt >= config.releaseMs;
    if (stable && cooldownOk && releaseOk) {
      state.lastSent = label;
      state.lastSentAt = now;
      return true;
    }
    return false;
  };

  const buildModelSpec = (model) => {
    const shape = model?.inputs?.[0]?.shape || [];
    const inputRank = shape.length;
    const featureSize = shape[shape.length - 1] || FEATURE_SETS.hands.size;
    const seqLen = inputRank >= 3 ? shape[shape.length - 2] : 1;
    const featureSet = resolveFeatureSet(featureSize);
    const sizeMismatch = featureSet.size !== featureSize;
    return {
      inputRank,
      featureSize,
      seqLen: seqLen || ASL_SEQUENCE_LENGTH,
      featureSet,
      sizeMismatch
    };
  };

  const loadAslModels = async () => {
    if (modelLoadRef.current) {
      return;
    }
    modelLoadRef.current = true;
    try {
      await tf.ready();

      if (!aslWordModelRef.current) {
        setAslWordStatus('loading');
        try {
          const model = await tf.loadLayersModel(WORD_MODEL_URL);
          const labelsResponse = await fetch(WORD_LABELS_URL);
          if (!labelsResponse.ok) {
            throw new Error('word labels not found');
          }
          const labelsData = await labelsResponse.json();
          aslWordModelRef.current = model;
          aslWordLabelsRef.current = parseLabels(labelsData);
          aslWordSpecRef.current = buildModelSpec(model);
          const featureKey = aslWordSpecRef.current.featureSet.key;
          const labelNote = aslWordLabelsRef.current.length ? '' : ' labels missing';
          const sizeNote = aslWordSpecRef.current.sizeMismatch ? ' feature mismatch' : '';
          setAslWordStatus(`loaded (${featureKey})${labelNote}${sizeNote}`);
        } catch (error) {
          setAslWordStatus('missing');
        }
      }

      if (!aslAlphaModelRef.current) {
        setAslAlphaStatus('loading');
        try {
          const model = await tf.loadLayersModel(ALPHA_MODEL_URL);
          const labelsResponse = await fetch(ALPHA_LABELS_URL);
          if (!labelsResponse.ok) {
            throw new Error('alpha labels not found');
          }
          const labelsData = await labelsResponse.json();
          aslAlphaModelRef.current = model;
          aslAlphaLabelsRef.current = parseLabels(labelsData);
          const alphaSpec =
            ASL_ALPHA_MODEL_TYPE === 'image'
              ? buildImageSpec(model)
              : { ...buildModelSpec(model), type: 'landmarks' };
          aslAlphaSpecRef.current = alphaSpec;
          const featureKey =
            alphaSpec.type === 'image'
              ? `${alphaSpec.width}x${alphaSpec.height}`
              : alphaSpec.featureSet.key;
          const labelNote = aslAlphaLabelsRef.current.length ? '' : ' labels missing';
          const sizeNote =
            alphaSpec.type === 'landmarks' && alphaSpec.sizeMismatch
              ? ' feature mismatch'
              : '';
          setAslAlphaStatus(
            `loaded (${alphaSpec.type} ${featureKey})${labelNote}${sizeNote}`
          );
        } catch (error) {
          setAslAlphaStatus('missing');
        }
      }
    } finally {
      modelLoadRef.current = false;
    }
  };

  const buildSequence = (sequenceRef, vector, maxLen) => {
    const next = sequenceRef.current.slice(-maxLen + 1);
    next.push(vector);
    sequenceRef.current = next;
    return next;
  };

  const padSequence = (sequence, targetLen, featureSize) => {
    const padded = sequence.slice(-targetLen);
    while (padded.length < targetLen) {
      padded.unshift(new Array(featureSize).fill(0));
    }
    return padded;
  };

  const predictWithModel = async (model, spec, sequence) => {
    if (!model || !spec) {
      return [];
    }
    let inputTensor;
    if (spec.inputRank === 2) {
      const lastFrame = sequence[sequence.length - 1] || new Array(spec.featureSize).fill(0);
      inputTensor = tf.tensor([lastFrame], [1, spec.featureSize]);
    } else {
      const targetLen = spec.seqLen || ASL_SEQUENCE_LENGTH;
      const padded = padSequence(sequence, targetLen, spec.featureSize);
      inputTensor = tf.tensor([padded], [1, targetLen, spec.featureSize]);
      if (spec.inputRank === 4) {
        inputTensor = inputTensor.expandDims(-1);
      }
    }

    const output = model.predict(inputTensor);
    const outputTensor = Array.isArray(output) ? output[0] : output;
    const scores = await outputTensor.data();
    tf.dispose([inputTensor, output]);
    return Array.from(scores);
  };

  const predictAlphaImage = async (model, spec, image, box) => {
    if (!model || !spec || !image || !box) {
      return [];
    }
    const boxes = tf.tensor2d([[box.minY, box.minX, box.maxY, box.maxX]]);
    const boxInd = tf.tensor1d([0], 'int32');
    const frame = tf.browser.fromPixels(image);
    const expanded = frame.expandDims(0);
    let crop = tf.image.cropAndResize(
      expanded,
      boxes,
      boxInd,
      [spec.height, spec.width]
    );
    if (spec.channels === 1) {
      crop = crop.mean(-1).expandDims(-1);
    }
    const normalized = crop.div(255);
    const output = model.predict(normalized);
    const outputTensor = Array.isArray(output) ? output[0] : output;
    const scores = await outputTensor.data();
    tf.dispose([boxes, boxInd, frame, expanded, crop, normalized, output]);
    return Array.from(scores);
  };

  const maybeAutoSendSentence = (now) => {
    if (!aslAutoSendRef.current) {
      return;
    }
    const text = gestureDraftRef.current.trim();
    const buffer = letterBufferRef.current.trim();
    const combined = [text, buffer].filter(Boolean).join(' ');
    if (!combined) {
      return;
    }
    const idleFor = now - lastSentenceActivityRef.current;
    if (idleFor < ASL_SENTENCE_PAUSE_MS) {
      return;
    }
    if (now - lastAutoSendRef.current < ASL_SENTENCE_PAUSE_MS) {
      return;
    }
    sendGestureText(combined, 'auto');
    setGestureDraft('');
    gestureDraftRef.current = '';
    setLetterBuffer('');
    letterBufferRef.current = '';
    lastAutoSendRef.current = now;
  };

  const runAslInference = async (results, now) => {
    const wordModel = aslWordModelRef.current;
    const alphaModel = aslAlphaModelRef.current;

    let wordPrediction = { label: '', score: 0 };
    let alphaPrediction = { label: '', score: 0 };

    if (wordModel && aslWordSpecRef.current) {
      const wordSpec = aslWordSpecRef.current;
      const wordFeatures = buildFeatureVector(results, wordSpec.featureSet);
      const wordSequence = buildSequence(
        aslWordSequenceRef,
        wordFeatures,
        Math.max(wordSpec.seqLen || ASL_SEQUENCE_LENGTH, ASL_SEQUENCE_LENGTH)
      );
      const wordScores = await predictWithModel(wordModel, wordSpec, wordSequence);
      wordPrediction = getTopPrediction(wordScores, aslWordLabelsRef.current);
      setAslWordPrediction(wordPrediction);

      const shouldCommitWord = updatePredictionState(
        aslWordStateRef,
        wordPrediction.label,
        wordPrediction.score,
        {
          stableMs: ASL_WORD_STABLE_MS,
          releaseMs: ASL_WORD_RELEASE_MS,
          cooldownMs: ASL_WORD_COOLDOWN_MS,
          minScore: ASL_WORD_CONFIDENCE
        },
        now
      );
      if (shouldCommitWord && aslAutoAppendRef.current) {
        appendSentenceWord(wordPrediction.label);
      }
    } else {
      setAslWordPrediction({ label: '', score: 0 });
    }

    if (
      alphaModel &&
      aslAlphaSpecRef.current &&
      aslAlphaSpecRef.current.type !== 'image'
    ) {
      const alphaSpec = aslAlphaSpecRef.current;
      const alphaFeatures = buildFeatureVector(results, alphaSpec.featureSet);
      const alphaSequence = buildSequence(
        aslAlphaSequenceRef,
        alphaFeatures,
        Math.max(alphaSpec.seqLen || ASL_SEQUENCE_LENGTH, ASL_SEQUENCE_LENGTH)
      );
      const alphaScores = await predictWithModel(
        alphaModel,
        alphaSpec,
        alphaSequence
      );
      alphaPrediction = getTopPrediction(alphaScores, aslAlphaLabelsRef.current);
      setAslAlphaPrediction(alphaPrediction);

      const shouldCommitLetter = updatePredictionState(
        aslAlphaStateRef,
        alphaPrediction.label,
        alphaPrediction.score,
        {
          stableMs: ASL_ALPHA_STABLE_MS,
          releaseMs: ASL_ALPHA_RELEASE_MS,
          cooldownMs: ASL_ALPHA_COOLDOWN_MS,
          minScore: ASL_ALPHA_CONFIDENCE
        },
        now
      );
      if (shouldCommitLetter && aslAutoAppendRef.current) {
        appendLetter(alphaPrediction.label);
      }
    } else {
      setAslAlphaPrediction({ label: '', score: 0 });
    }

    if (letterBufferRef.current && now - lastLetterAtRef.current >= ASL_LETTER_PAUSE_MS) {
      commitLetterBuffer();
    }

    if (!wordModel && !alphaModel) {
      setGestureSummary('ASL models not loaded');
    } else {
      const wordText = wordModel
        ? wordPrediction.label
          ? `${wordPrediction.label} ${Math.round(wordPrediction.score * 100)}%`
          : '...'
        : 'model missing';
      const alphaText = alphaModel
        ? alphaPrediction.label
          ? `${alphaPrediction.label} ${Math.round(alphaPrediction.score * 100)}%`
          : '...'
        : 'model missing';
      setGestureSummary(`Word: ${wordText} | Fingerspell: ${alphaText}`);
    }
  };

  const runAlphaImageInference = async (results, now) => {
    const alphaModel = aslAlphaModelRef.current;
    const alphaSpec = aslAlphaSpecRef.current;
    if (!alphaModel || !alphaSpec || alphaSpec.type !== 'image') {
      setAslAlphaPrediction({ label: '', score: 0 });
      return;
    }

    if (now - lastAlphaImageInferenceRef.current < ASL_ALPHA_IMAGE_INTERVAL_MS) {
      return;
    }
    if (alphaImageInFlightRef.current) {
      return;
    }
    lastAlphaImageInferenceRef.current = now;

    const handLandmarks = chooseHandLandmarks(
      results.leftHandLandmarks,
      results.rightHandLandmarks
    );
    const box = getHandBox(handLandmarks);
    if (!box || !results.image) {
      setAslAlphaPrediction({ label: '', score: 0 });
      return;
    }

    alphaImageInFlightRef.current = true;
    try {
      const scores = await predictAlphaImage(
        alphaModel,
        alphaSpec,
        results.image,
        box
      );
      const prediction = getTopPrediction(scores, aslAlphaLabelsRef.current);
      const label = prediction.label
        ? String(prediction.label).trim().toUpperCase()
        : '';
      const score = prediction.score || 0;
      setAslAlphaPrediction({ label, score });

      const shouldCommitLetter = updatePredictionState(
        aslAlphaStateRef,
        label,
        score,
        {
          stableMs: ASL_ALPHA_STABLE_MS,
          releaseMs: ASL_ALPHA_RELEASE_MS,
          cooldownMs: ASL_ALPHA_COOLDOWN_MS,
          minScore: ASL_ALPHA_CONFIDENCE
        },
        now
      );
      if (shouldCommitLetter && aslAutoAppendRef.current) {
        appendLetter(label);
      }

      if (letterBufferRef.current && now - lastLetterAtRef.current >= ASL_LETTER_PAUSE_MS) {
        commitLetterBuffer();
      }

      if (aslModeRef.current === 'hybrid') {
        const serverText = formatPrediction(
          aslServerPredictionRef.current?.label,
          aslServerPredictionRef.current?.score
        );
        const alphaText = formatPrediction(label, score);
        setGestureSummary(`Server: ${serverText} | Alpha: ${alphaText}`);
      } else {
        setGestureSummary(`Recognition: ${formatPrediction(label, score)}`);
      }
    } finally {
      alphaImageInFlightRef.current = false;
    }
  };

  const runAlphaServerInference = async (results, now) => {
    if (now - lastAlphaServerInferenceRef.current < ASL_ALPHA_SERVER_INTERVAL_MS) {
      return;
    }
    if (alphaServerInFlightRef.current) {
      return;
    }

    const handLandmarks = chooseHandLandmarks(
      results.leftHandLandmarks,
      results.rightHandLandmarks
    );

    if (!handLandmarks || handLandmarks.length < 21) {
      setAslAlphaPrediction({ label: '', score: 0 });
      return;
    }

    lastAlphaServerInferenceRef.current = now;
    alphaServerInFlightRef.current = true;

    try {
      const response = await fetch(`${ASL_SERVER_URL}/asl/predict_alpha`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ landmarks: handLandmarks })
      });

      if (!response.ok) {
        throw new Error('Server error');
      }

      const data = await response.json();
      const rawLabel = data.label ? String(data.label).trim() : '';
      const score = data.score || 0;
      const label = normalizeHandLabel(rawLabel);
      setAslAlphaPrediction({ label, score });

      const shouldCommit = updatePredictionState(
        aslAlphaStateRef,
        rawLabel,
        score,
        {
          stableMs: ASL_ALPHA_STABLE_MS,
          releaseMs: ASL_ALPHA_RELEASE_MS,
          cooldownMs: ASL_ALPHA_COOLDOWN_MS,
          minScore: ASL_ALPHA_CONFIDENCE
        },
        now
      );

      if (shouldCommit && aslAutoAppendRef.current) {
        if (rawLabel === 'space') {
          commitLetterBuffer();
        } else if (rawLabel === 'del') {
          deleteLastLetter();
        } else if (rawLabel === 'nothing') {
          // No-operation
        } else if (rawLabel.length > 1) {
          commitLetterBuffer();
          appendSentenceWord(label);
        } else {
          appendLetter(label);
        }
      }

      if (letterBufferRef.current && now - lastLetterAtRef.current >= ASL_LETTER_PAUSE_MS) {
        commitLetterBuffer();
      }

      if (aslModeRef.current === 'hybrid') {
        const serverText = formatPrediction(
          aslServerPredictionRef.current?.label,
          aslServerPredictionRef.current?.score
        );
        const alphaText = formatPrediction(label, score);
        setGestureSummary(`Server: ${serverText} | Recognition: ${alphaText}`);
      } else {
        setGestureSummary(`Recognition: ${formatPrediction(label, score)}`);
      }
    } catch (err) {
      setAslAlphaPrediction({ label: '', score: 0 });
    } finally {
      alphaServerInFlightRef.current = false;
    }
  };

  const resetAslRuntime = (summary) => {
    aslWordSequenceRef.current = [];
    aslAlphaSequenceRef.current = [];
    aslWordStateRef.current = {
      candidate: null,
      candidateSince: 0,
      lastSent: null,
      lastSentAt: 0,
      lastClearAt: 0
    };
    aslAlphaStateRef.current = {
      candidate: null,
      candidateSince: 0,
      lastSent: null,
      lastSentAt: 0,
      lastClearAt: 0
    };
    lastSentenceActivityRef.current = Date.now();
    lastAutoSendRef.current = 0;
    lastInferenceRef.current = 0;
    inferenceInFlightRef.current = false;
    lastLetterAtRef.current = 0;
    lastAlphaImageInferenceRef.current = 0;
    alphaImageInFlightRef.current = false;
    lastAlphaServerInferenceRef.current = 0;
    alphaServerInFlightRef.current = false;
    letterBufferRef.current = '';
    setLetterBuffer('');
    setAslWordPrediction({ label: '', score: 0 });
    setAslAlphaPrediction({ label: '', score: 0 });
    setAslServerPrediction({ label: '', score: 0 });
    if (summary) {
      setGestureSummary(summary);
    }
  };

  const startGestures = async () => {
    if (gestureEnabled) {
      return;
    }

    const stream = await ensureLocalStream();
    if (!stream) {
      return;
    }

    if (aslModeRef.current === 'client' || aslModeRef.current === 'hybrid') {
      setGestureSummary('Loading client-side models...');
      await loadAslModels();
    }
    if (aslModeRef.current === 'server' || aslModeRef.current === 'hybrid') {
      setGestureSummary('Probing AI server...');
      await probeAslServer();
    }
    setGestureSummary('Initializing MediaPipe...');

    let holistic;
    try {
      const SelectedHolistic = window.Holistic || Holistic;
      if (!SelectedHolistic) {
        throw new Error('Holistic constructor not found. CDN scripts may have failed to load.');
      }
      holistic = new SelectedHolistic({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      });
    } catch (err) {
      console.error('MediaPipe Init Error:', err);
      setGestureSummary(`MediaPipe Error: ${err.message}`);
      return;
    }

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6
    });

    holistic.onResults((results) => {
      const canvas = overlayCanvasRef.current;
      if (!canvas || !localVideoRef.current) {
        return;
      }
      const ctx = canvas.getContext('2d');
      const width = localVideoRef.current.videoWidth || 640;
      const height = localVideoRef.current.videoHeight || 480;
      canvas.width = width;
      canvas.height = height;

      if (gestureSummaryRef.current === 'Initializing MediaPipe...') {
        setGestureSummary('MediaPipe ready');
      }

      ctx.save();
      ctx.clearRect(0, 0, width, height);

      if (results.leftHandLandmarks) {
        drawConnectors(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, {
          color: '#f36f45',
          lineWidth: 2
        });
        drawLandmarks(ctx, results.leftHandLandmarks, {
          color: '#2a9d8f',
          lineWidth: 1
        });
      }

      if (results.rightHandLandmarks) {
        drawConnectors(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, {
          color: '#f36f45',
          lineWidth: 2
        });
        drawLandmarks(ctx, results.rightHandLandmarks, {
          color: '#2a9d8f',
          lineWidth: 1
        });
      }

      // Landmark-based inference (Alphabet + Common Words)
      const now = Date.now();
      if (ASL_ALPHA_MODEL_TYPE === 'server') {
        runAlphaServerInference(results, now);
      } else if (aslModeRef.current === 'client') {
        const canRunLandmark =
          now - lastInferenceRef.current >= ASL_INFERENCE_INTERVAL_MS &&
          !inferenceInFlightRef.current;
        if (canRunLandmark) {
          lastInferenceRef.current = now;
          inferenceInFlightRef.current = true;
          runAslInference(results, now).finally(() => {
            inferenceInFlightRef.current = false;
          });
        }
      } else if (aslAlphaSpecRef.current?.type === 'image') {
        runAlphaImageInference(results, now);
      }

      ctx.restore();
      maybeAutoSendSentence(now);
    });

    holisticRef.current = holistic;
    setGestureEnabled(true);
    gestureEnabledRef.current = true;
    if (aslModeRef.current === 'server') {
      resetAslRuntime('ASL server ready');
    } else if (aslModeRef.current === 'hybrid') {
      resetAslRuntime('ASL hybrid ready');
    } else {
      resetAslRuntime('ASL capture running');
    }

    let active = true;
    const loop = async () => {
      if (!active || !holisticRef.current || !localVideoRef.current) {
        return;
      }
      await holisticRef.current.send({ image: localVideoRef.current });
      requestAnimationFrame(loop);
    };

    gestureLoopRef.current = () => {
      active = false;
    };

    loop();
  };

  const stopGestures = () => {
    stopServerAutoCapture();
    stopServerRecording();
    if (gestureLoopRef.current) {
      gestureLoopRef.current();
      gestureLoopRef.current = null;
    }
    if (holisticRef.current) {
      holisticRef.current.close();
      holisticRef.current = null;
    }
    setGestureEnabled(false);
    gestureEnabledRef.current = false;
    resetAslRuntime('ASL capture stopped');

    const canvas = overlayCanvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  const shareLocation = (position) => {
    const coords = position.coords || {};
    sendWs({
      type: 'location',
      coords: {
        lat: coords.latitude,
        lng: coords.longitude,
        accuracy: coords.accuracy
      },
      timestamp: position.timestamp
    });
    setLocationStatus('location sent');
  };

  const shareLocationOnce = () => {
    if (!navigator.geolocation) {
      setLocationStatus('geolocation unavailable');
      return;
    }
    setLocationStatus('locating...');
    navigator.geolocation.getCurrentPosition(shareLocation, () => {
      setLocationStatus('location permission denied');
    });
  };

  const startLiveLocation = () => {
    if (!navigator.geolocation) {
      setLocationStatus('geolocation unavailable');
      return;
    }
    if (locationSharing) {
      return;
    }
    setLocationStatus('sharing live location');
    const watchId = navigator.geolocation.watchPosition(
      shareLocation,
      () => {
        setLocationStatus('unable to share location');
        stopLiveLocation();
      },
      { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
    );
    locationWatchRef.current = watchId;
    setLocationSharing(true);
  };

  const stopLiveLocation = () => {
    if (locationWatchRef.current != null) {
      navigator.geolocation.clearWatch(locationWatchRef.current);
    }
    locationWatchRef.current = null;
    setLocationSharing(false);
    setLocationStatus('live sharing stopped');
  };

  const startTimer = () => {
    const minutes = Number(timerMinutes);
    if (!callActive) {
      setStatus('start a call first');
      return;
    }
    if (!Number.isFinite(minutes) || minutes <= 0) {
      setStatus('enter a valid duration');
      return;
    }

    clearTimer();
    const duration = minutes * 60 * 1000;
    setTimerActive(true);
    setTimeRemaining(duration);

    timerTimeoutRef.current = setTimeout(() => {
      endCall();
      setTimerActive(false);
    }, duration);

    timerIntervalRef.current = setInterval(() => {
      setTimeRemaining((prev) => (prev == null ? prev : prev - 1000));
    }, 1000);
  };

  useEffect(() => {
    return () => {
      stopGestures();
      stopLiveLocation();
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pcsRef.current) {
        pcsRef.current.forEach(pc => pc.close());
      }
      stopLocalStream();
      clearTimer();
      if (roomHintTimeoutRef.current) {
        clearTimeout(roomHintTimeoutRef.current);
        roomHintTimeoutRef.current = null;
      }
    };
  }, []);

  const voiceForSpeak = voices.find(
    (voice) => voice.voiceURI === selectedVoice
  );

  const remoteMapUrl = remoteLocation?.lat
    ? `https://www.google.com/maps?q=${remoteLocation.lat},${remoteLocation.lng}`
    : null;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <img src="/logo.png" alt="S MEET logo" className="logo" />
          <h1 style={{ textTransform: 'uppercase' }}>S MEET</h1>
        </div>
        <p>Real-time video calling with gesture AI, live TTS, and location sharing.</p>
        <div className="status-row">
          <span className="pill">
            {connected ? 'Signaling connected' : 'Signaling offline'}
          </span>
          <span className={`pill ${connected && isInitiator ? '' : 'warning'}`}>
            {connected ? (isInitiator ? 'Room host' : 'Room guest') : 'Not in room'}
          </span>
          <span className="pill warning">Status: {status}</span>
        </div>
      </header>

      {connected ? (
        <>
          <div className="room-bar">
            <div>
              <div className="room-id">Room: {roomId || 'unknown'}</div>
              <div className="status-text">
                Share your room id or join link with the other user.
              </div>
            </div>
            <div className="room-actions">
              <button
                className="secondary"
                onClick={copyRoomId}
                disabled={!roomId.trim()}
              >
                Copy room id
              </button>
              <button
                className="secondary"
                onClick={copyRoomLink}
                disabled={!roomId.trim()}
              >
                Copy join link
              </button>
              <button className="secondary" onClick={disconnectSocket}>
                Leave room
              </button>
            </div>
          </div>
          {roomHint ? <div className="status-text room-hint">{roomHint}</div> : null}
          <div className="layout">
            <div className="hero-grid">
              <section className="panel">
                <div className="panel-head">
                  <div>
                    <h2>Camera and call</h2>
                    <p className="panel-subtitle">
                      Preview your devices and start the call when ready.
                    </p>
                  </div>
                  <span className="step">Step 2</span>
                </div>
                <div className="video-grid">
                  <div className="video-card">
                    <h3>You</h3>
                    <div className="video-shell">
                      <video
                        className="mirrored"
                        ref={localVideoRef}
                        autoPlay
                        muted
                        playsInline
                      />
                      <canvas className="overlay" ref={overlayCanvasRef} />
                      {sttEnabled && sttTranscript ? (
                        <div className="captions-overlay">{sttTranscript}</div>
                      ) : null}
                    </div>
                  </div>
                  <div className="video-card" style={{ flex: '2 1 400px', display: 'flex', flexDirection: 'column' }}>
                    <h3>Remote ({remoteStreamsRef.current.size})</h3>
                    <div className="video-shell" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', padding: '8px', background: 'transparent' }}>
                      {remoteStreamsRef.current.size > 0 ? (
                        Array.from(remoteStreamsRef.current.entries()).map(([peerId, stream]) => (
                          <div key={peerId} style={{ flex: '1 1 45%', minWidth: '200px', position: 'relative' }}>
                            <video 
                              autoPlay 
                              playsInline 
                              ref={(el) => { if (el) attachStreamToVideo(el, stream); }} 
                              style={{ width: '100%', height: '100%', objectFit: 'cover', borderRadius: '12px' }} 
                            />
                            {remoteSttText && (
                              <div className="captions-overlay">{remoteSttText}</div>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="video-placeholder flex-col" style={{ width: '100%', minHeight: '300px' }}>
                          <div className="icon large">📡</div>
                          <div>Waiting for video</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                <div className="controls">
                  <div className="button-row">
                    <button onClick={startCall} disabled={!connected}>
                      Start call
                    </button>
                    <button
                      className="warning"
                      onClick={() => endCall(true)}
                      disabled={!localStream && !callActive}
                    >
                      End call
                    </button>
                    <button
                      className="secondary"
                      onClick={() => ensureLocalStream()}
                    >
                      Preview camera
                    </button>
                  </div>
                  <div className="button-row">
                    <button
                      className="secondary"
                      onClick={() => toggleTrack('audio')}
                      disabled={!localStream}
                    >
                      {micEnabled ? 'Mute mic' : 'Unmute mic'}
                    </button>
                    <button
                      className="secondary"
                      onClick={() => toggleTrack('video')}
                      disabled={!localStream}
                    >
                      {camEnabled ? 'Hide camera' : 'Show camera'}
                    </button>
                    <button
                      className={`secondary ${sttEnabled ? '' : 'ghost'}`}
                      onClick={() => setSttEnabled(!sttEnabled)}
                      disabled={!sttSupported}
                    >
                      {sttEnabled ? 'Disable CC' : 'Enable CC'}
                    </button>
                    <button
                      className="secondary"
                      onClick={toggleFullscreen}
                      disabled={!localStream && !callActive}
                    >
                      Fullscreen
                    </button>
                  </div>
                  <div className="status-text">Device: {deviceStatus}</div>
                  {!sttSupported && <div className="status-text warning">Speech-to-Text not supported in this browser.</div>}
                  {deviceStatus === 'requires https or localhost' ? (
                    <div className="status-text">
                      Open the app on http://localhost:5173 or use https.
                    </div>
                  ) : null}
                  <div className="status-text">
                    Call status: {callActive ? 'live' : 'idle'}
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-head">
                  <div>
                    <h2>ASL recognition</h2>
                    <p className="panel-subtitle">
                      Cloud-powered ASL alphabet and common sign recognition.
                    </p>
                  </div>
                  <span className="step">Step 3</span>
                </div>
                <div className="controls">
                  <span className="tag">Detected: {gestureSummary}</span>
                  <div className="field">
                    <label htmlFor="asl-mode">ASL mode</label>
                    <select
                      id="asl-mode"
                      value={aslMode}
                      onChange={handleAslModeChange}
                    >
                      <option value="server">Server (Letters & Words)</option>
                    </select>
                  </div>
                  <div className="button-row">
                    {aslMode === 'server' || aslMode === 'hybrid' ? (
                      <button className="ghost" onClick={toggleAutoCapture}>
                        {aslAutoCapture ? 'Auto capture on' : 'Auto capture off'}
                      </button>
                    ) : null}
                    <button
                      className="ghost"
                      onClick={() => setAslAutoAppend((prev) => !prev)}
                    >
                      {aslAutoAppend ? 'Auto append on' : 'Auto append off'}
                    </button>
                    <button
                      className="ghost"
                      onClick={() => setAslAutoSend((prev) => !prev)}
                    >
                      {aslAutoSend ? 'Auto send on' : 'Auto send off'}
                    </button>
                  </div>
                  <div className="button-row">
                    <button onClick={startGestures} disabled={gestureEnabled}>
                      Start capture
                    </button>
                    <button
                      className="secondary"
                      onClick={stopGestures}
                      disabled={!gestureEnabled}
                    >
                      Stop capture
                    </button>
                    {aslMode === 'server' ? (
                      <button
                        className="secondary"
                        onClick={captureServerClip}
                        disabled={!gestureEnabled || aslServerBusy}
                      >
                        {aslServerBusy ? 'Capturing...' : 'Capture sign'}
                      </button>
                    ) : null}
                    {aslMode === 'hybrid' ? (
                      <>
                        <button
                          className="secondary"
                          onClick={captureServerClip}
                          disabled={!gestureEnabled || aslServerBusy}
                        >
                          {aslServerBusy ? 'Capturing...' : 'Capture sign'}
                        </button>
                        <button className="secondary" onClick={loadAslModels}>
                          Load alphabet
                        </button>
                      </>
                    ) : null}
                    {aslMode === 'client' ? (
                      <button className="secondary" onClick={loadAslModels}>
                        Load models
                      </button>
                    ) : null}
                  </div>
                  {aslMode === 'server' ? (
                    <>
                      <div className="status-text">Server: {aslServerStatus}</div>
                      <div className="status-text">
                        Model status: {aslAlphaStatus}
                      </div>
                      <div className="status-text">
                        Prediction: {aslServerPrediction.label || '...'}{' '}
                        {aslServerPrediction.score
                          ? `${Math.round(aslServerPrediction.score * 100)}%`
                          : ''}
                      </div>
                      <div className="button-row">
                        <button className="ghost" onClick={probeAslServer}>
                          Check server
                        </button>
                      </div>
                    </>
                  ) : null}

                  <div className="field">
                    <label htmlFor="gesture-draft">ASL sentence</label>
                    <textarea
                      id="gesture-draft"
                      value={gestureDraft}
                      onChange={(event) => setGestureDraft(event.target.value)}
                      placeholder="Build a sentence with ASL..."
                    />
                  </div>
                  {aslMode !== 'server' ? (
                    <div className="field">
                      <label htmlFor="letter-buffer">Finger-spell buffer</label>
                      <input
                        id="letter-buffer"
                        value={letterBuffer}
                        readOnly
                        placeholder="Letters captured from finger-spelling"
                      />
                    </div>
                  ) : null}
                  <div className="button-row">
                    <button
                      className="ghost"
                      onClick={sendGestureDraft}
                      disabled={
                        !connected ||
                        (!gestureDraft.trim() && !letterBuffer.trim())
                      }
                    >
                      Send sentence
                    </button>
                    {aslMode !== 'server' ? (
                      <button
                        className="secondary"
                        onClick={commitLetterBuffer}
                        disabled={!letterBuffer.trim()}
                      >
                        Commit buffer
                      </button>
                    ) : null}
                    <button
                      className="secondary"
                      onClick={clearGestureDraft}
                      disabled={!gestureDraft.trim() && !letterBuffer.trim()}
                    >
                      Clear
                    </button>
                  </div>
                  <div className="status-text">
                    {aslMode === 'server'
                      ? 'Capture a short clip per sign. Enable auto capture for continuous signing.'
                      : aslMode === 'hybrid'
                        ? 'Capture clips for words and finger-spell letters in parallel.'
                        : 'Hold a sign briefly. Pause to commit finger-spelled letters.'}
                  </div>
                </div>
              </section>
            </div>

            <aside className="controls">
              <section className="panel">
                <h2>Text and voice</h2>
                <div className="field">
                  <label htmlFor="voice">Preferred voice</label>
                  <select
                    id="voice"
                    value={selectedVoice}
                    onChange={(event) => setSelectedVoice(event.target.value)}
                  >
                    <option value="">Auto / default voice</option>
                    {voiceOptions.map((voice) => (
                      <option key={voice.value} value={voice.value}>
                        {voice.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="button-row">
                  <button
                    className="ghost"
                    onClick={() => setAutoSpeak((prev) => !prev)}
                  >
                    {autoSpeak ? 'Auto speak on' : 'Auto speak off'}
                  </button>
                </div>
                {!ttsSupported ? (
                  <div className="status-text">
                    Speech synthesis is not supported in this browser.
                  </div>
                ) : null}
                <div className="field">
                  <label htmlFor="message">Send a message</label>
                  <div className="split">
                    <input
                      id="message"
                      value={messageInput}
                      onChange={(event) => setMessageInput(event.target.value)}
                      placeholder="Type a message"
                      onKeyDown={(event) => {
                        if (event.key === 'Enter') {
                          sendChat();
                        }
                      }}
                    />
                    <button onClick={sendChat} disabled={!connected}>
                      Send
                    </button>
                  </div>
                </div>
                <div className="chat-list">
                  {messages.length === 0 ? (
                    <div className="status-text">No messages yet.</div>
                  ) : (
                    messages.map((message) => (
                      <div
                        key={message.id}
                        className={`message ${message.self ? 'self' : ''}`}
                      >
                        <div className="meta">
                          {message.type} - {message.from}
                        </div>
                        <div className="text">{message.text}</div>
                        {ttsSupported ? (
                          <div className="button-row">
                            <button
                              className="secondary"
                              onClick={() => speak(message.text, voiceForSpeak)}
                            >
                              Speak
                            </button>
                          </div>
                        ) : null}
                      </div>
                    ))
                  )}
                </div>
              </section>

              <section className="panel">
                <h2>Live location</h2>
                <div className="button-row">
                  <button onClick={shareLocationOnce} disabled={!connected}>
                    Share once
                  </button>
                  <button
                    className="secondary"
                    onClick={locationSharing ? stopLiveLocation : startLiveLocation}
                    disabled={!connected}
                  >
                    {locationSharing ? 'Stop live' : 'Start live'}
                  </button>
                </div>
                <div className="location-card">Status: {locationStatus}</div>
                {remoteLocation ? (
                  <div className="location-card">
                    Remote location from {remoteLocation.from || 'peer'}: 
                    {remoteLocation.lat?.toFixed(4)}, {remoteLocation.lng?.toFixed(4)}
                    {remoteLocation.accuracy
                      ? ` (+/-${Math.round(remoteLocation.accuracy)}m)`
                      : ''}
                    {remoteMapUrl ? (
                      <div>
                        <a href={remoteMapUrl} target="_blank" rel="noreferrer">
                          Open map
                        </a>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>

              <section className="panel">
                <h2>Auto call timer</h2>
                <div className="field">
                  <label htmlFor="timer">Minutes</label>
                  <input
                    id="timer"
                    type="number"
                    min="1"
                    value={timerMinutes}
                    onChange={(event) => setTimerMinutes(event.target.value)}
                  />
                </div>
                <div className="button-row">
                  <button onClick={startTimer} disabled={!callActive}>
                    Start timer
                  </button>
                  <button
                    className="secondary"
                    onClick={clearTimer}
                    disabled={!timerActive}
                  >
                    Clear timer
                  </button>
                </div>
                {timerActive ? (
                  <div className="timer">
                    Time remaining: {formatTimer(timeRemaining)}
                  </div>
                ) : null}
              </section>
            </aside>
          </div>
        </>
      ) : (
        <div className="lobby">
          <section className="panel lobby-hero">
            <span className="tag">Welcome</span>
            <h2>Start with a room.</h2>
            <p>
              Create a private room and share the join link. Once connected, you
              can preview the camera and start gestures.
            </p>
            <div className="lobby-list">
              <div>Create a room to get a shareable code.</div>
              <div>Join an existing room to connect instantly.</div>
              <div>Then open the call workspace and test gestures.</div>
            </div>
            <div className="status-text">Signaling: {SIGNALING_URL}</div>
          </section>

          <section className="panel lobby-card">
            <div className="panel-head">
              <div>
                <h2>Create or join</h2>
                <p className="panel-subtitle">
                  Rooms are temporary and only exist while you are connected.
                </p>
              </div>
              <span className="step">Step 1</span>
            </div>
            <div className="controls">
              <div className="field">
                <label htmlFor="name">Your name</label>
                <input
                  id="name"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Add your display name"
                />
              </div>
              <div className="field">
                <label htmlFor="room">Room ID</label>
                <input
                  id="room"
                  value={roomId}
                  onChange={(event) => setRoomId(event.target.value)}
                  placeholder="e.g. team-sync"
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      connectSocket();
                    }
                  }}
                />
              </div>
              <div className="button-row">
                <button onClick={createRoom}>Create room</button>
                <button onClick={() => connectSocket()}>Join room</button>
              </div>
              <div className="button-row">
                <button
                  className="secondary"
                  onClick={copyRoomId}
                  disabled={!roomId.trim()}
                >
                  Copy room id
                </button>
                <button
                  className="secondary"
                  onClick={copyRoomLink}
                  disabled={!roomId.trim()}
                >
                  Copy join link
                </button>
              </div>
              {roomHint ? <div className="status-text">{roomHint}</div> : null}
              <div className="status-text">Client id: {clientId || 'pending'}</div>
            </div>
          </section>
        </div>
      )}

      {/* Immersive Fullscreen Overlay */}
      {isFullscreen && (
        <div className="fullscreen-overlay">
          <div className="fs-main-content">
            <div className="fs-video-container">
              {/* Remote Participant (Main) */}
              <div className="fs-remote-video">
                <video 
                  ref={fsRemoteVideoRef} 
                  autoPlay 
                  playsInline 
                />
                
                {/* Captions Area */}
                <div className="fs-captions-container">
                  {remoteSttText && (
                    <div className="fs-captions remote">{remoteSttText}</div>
                  )}
                  {sttEnabled && sttTranscript && (
                    <div className="fs-captions local">{sttTranscript}</div>
                  )}
                </div>
              </div>
              
              {/* Local Participant (PiP) */}
              <div className="fs-local-video">
                <video 
                  className="mirrored"
                  ref={fsLocalVideoRef} 
                  autoPlay 
                  muted 
                  playsInline 
                />
              </div>
            </div>

            {/* Floating Chat Sidebar */}
            <div className="fs-chat-sidebar">
              <div className="fs-chat-head">
                <h3>Chat</h3>
                <button className="fs-icon-btn" onClick={toggleFullscreen}>✕</button>
              </div>
              <div className="fs-chat-messages">
                {messages.map((msg) => (
                  <div key={msg.id} className={`fs-message ${msg.self ? 'self' : ''}`}>
                    <div className="fs-message-meta">{msg.from}</div>
                    <div className="fs-message-text">{msg.text}</div>
                  </div>
                ))}
              </div>
              <div className="fs-chat-input">
                <input 
                  type="text" 
                  placeholder="Send a message..." 
                  value={messageInput}
                  onChange={(e) => setMessageInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendChat()}
                />
                <button onClick={sendChat}>Send</button>
              </div>
            </div>
          </div>

          {/* Immersive Control Bar */}
          <div className="fs-controls-bar">
            <button 
              className={`fs-btn ${micEnabled ? '' : 'off'}`} 
              onClick={() => toggleTrack('audio')}
            >
              {micEnabled ? 'Mute' : 'Unmute'}
            </button>
            <button 
              className={`fs-btn ${camEnabled ? '' : 'off'}`} 
              onClick={() => toggleTrack('video')}
            >
              {camEnabled ? 'No Cam' : 'Cam'}
            </button>
            <button 
              className="fs-btn exit" 
              onClick={toggleFullscreen}
            >
              Exit Fullscreen
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
