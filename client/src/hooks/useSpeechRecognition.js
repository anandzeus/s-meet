import { useState, useEffect, useRef, useCallback } from 'react';

export function useSpeechRecognition() {
  const [supported, setSupported] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  
  const recognitionRef = useRef(null);
  const listeningRef = useRef(false);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (SpeechRecognition) {
      console.log('[STT] SpeechRecognition API is supported.');
      setSupported(true);
      const recognition = new SpeechRecognition();
      // Safari often struggles with continuous = true, but we try it
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.lang = 'en-US';

      recognition.onresult = (event) => {
        let currentTranscript = '';
        for (let i = 0; i < event.results.length; i++) {
          const transcriptPiece = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            currentTranscript += transcriptPiece + ' ';
          } else {
            currentTranscript += transcriptPiece;
          }
        }
        console.log('[STT] Transcript Update:', currentTranscript.trim());
        setTranscript(currentTranscript.trim());
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
          setListening(false);
          listeningRef.current = false;
        }
      };

      recognition.onend = () => {
        console.log('[STT] Session ended. Auto-restart?', listeningRef.current);
        if (listeningRef.current) {
          // Add a slight delay before restarting to appease Safari
          setTimeout(() => {
            if (listeningRef.current && recognitionRef.current) {
               try {
                 recognitionRef.current.start();
               } catch (err) {
                 console.error('[STT] Auto-restart failed:', err);
               }
            }
          }, 400);
        } else {
          setListening(false);
        }
      };

      recognitionRef.current = recognition;
    } else {
      setSupported(false);
      console.warn('[STT] Speech Recognition is not supported in this browser.');
    }

    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
    };
  }, []);

  const startListening = useCallback(() => {
    if (!recognitionRef.current) return;
    try {
      console.log('[STT] Starting recognition...');
      recognitionRef.current.start();
      setListening(true);
      listeningRef.current = true;
      setTranscript('');
    } catch (err) {
      console.error('[STT] Failed to start recognition:', err);
    }
  }, []);

  const stopListening = useCallback(() => {
    if (!recognitionRef.current) return;
    console.log('[STT] Stopping recognition...');
    listeningRef.current = false;
    setListening(false);
    recognitionRef.current.stop();
  }, []);

  const clearTranscript = useCallback(() => {
    setTranscript('');
  }, []);

  return {
    supported,
    listening,
    transcript,
    startListening,
    stopListening,
    clearTranscript
  };
}
