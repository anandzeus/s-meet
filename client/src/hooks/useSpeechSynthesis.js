import { useEffect, useState } from 'react';

export const useSpeechSynthesis = () => {
  const [voices, setVoices] = useState([]);

  useEffect(() => {
    const loadVoices = () => {
      if (!window.speechSynthesis) {
        setVoices([]);
        return;
      }
      setVoices(window.speechSynthesis.getVoices());
    };

    loadVoices();

    if (window.speechSynthesis) {
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }

    return () => {
      if (window.speechSynthesis) {
        window.speechSynthesis.onvoiceschanged = null;
      }
    };
  }, []);

  const speak = (text, voice) => {
    if (!window.speechSynthesis || !text) {
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    if (voice) {
      utterance.voice = voice;
    }
    window.speechSynthesis.speak(utterance);
  };

  return {
    voices,
    speak,
    supported: Boolean(window.speechSynthesis)
  };
};