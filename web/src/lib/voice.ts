// Local speech for the pages: what the server can do (GET /voice), the "Speak replies" setting remembered per browser, and
// playing text aloud through POST /voice/speak. Chat uses all of it; any page that shows a model's text can reuse it.
// Both engines run on this machine and are installed by `make voice-setup` (docs/voice.md); the app never installs them.

import { useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, apiFetch, unwrap } from "@/lib/api";
import type { components } from "@/lib/api-schema";

export type VoiceStatus = components["schemas"]["VoiceStatus"];

const SPEAK_KEY = "voice.speakReplies";
const TTS_VOICE_KEY = "voice.ttsVoice";

export const TTS_VOICES = [
  { id: "lessac", label: "Lessac · American English · high quality" },
  { id: "ryan", label: "Ryan · American English · high quality" },
  { id: "amy", label: "Amy · American English · medium quality" },
] as const;
export type TtsVoice = (typeof TTS_VOICES)[number]["id"];
export const DEFAULT_TTS_VOICE: TtsVoice = "lessac";

function storedTtsVoice(): TtsVoice {
  try {
    const stored = localStorage.getItem(TTS_VOICE_KEY);
    return TTS_VOICES.some(({ id }) => id === stored) ? (stored as TtsVoice) : DEFAULT_TTS_VOICE;
  } catch {
    return DEFAULT_TTS_VOICE;
  }
}

/** The selected spoken-reply voice, remembered in this browser. */
export function useTtsVoice(): [TtsVoice, (voice: TtsVoice) => void] {
  const [voice, setVoice] = useState<TtsVoice>(DEFAULT_TTS_VOICE);
  useEffect(() => setVoice(storedTtsVoice()), []);
  const choose = useCallback((next: TtsVoice) => {
    setVoice(next);
    try {
      localStorage.setItem(TTS_VOICE_KEY, next);
    } catch {
      // The setting then lasts for the visit.
    }
  }, []);
  return [voice, choose];
}

/** The engine's own sentence when it cannot run ("... run `make voice-setup`"), "" when it is installed, or a sentence when the server does not list it. */
export function engineIssue(status: VoiceStatus | undefined, engine: "stt" | "tts"): string {
  const found = status?.engines.find((e) => e.engine === engine);
  if (!found) return `This server does not report a ${engine === "stt" ? "speech to text" : "text to speech"} engine. Update it and run \`make voice-setup\`.`;
  if (found.installed) return "";
  return found.missing || `${found.label} is not installed. Run \`make voice-setup\`.`;
}

/** A sentence for a failed voice call: the server's `detail` when it sent one, else what went wrong. */
export function voiceMessage(error: unknown): string {
  if (error instanceof ApiError) {
    try {
      const detail = (JSON.parse(error.body) as { detail?: unknown }).detail;
      if (typeof detail === "string" && detail) return detail;
    } catch {
      // not JSON: fall through to the status
    }
    return `The voice engine answered ${error.status}${error.body ? `: ${error.body}` : ""}`;
  }
  return error instanceof Error ? error.message : "Something went wrong with the voice engine.";
}

/** What `GET /voice` says about each engine. Re-read on demand (after a `make voice-setup`) with `refetch`. */
export function useVoiceStatus() {
  return useQuery({ queryKey: ["voice", "status"], queryFn: () => unwrap(api.GET("/voice")), staleTime: 30_000 });
}

/** Send a 16 kHz mono 16-bit WAV to the speech to text engine and get the words back. */
export async function transcribe(wav: ArrayBuffer, signal?: AbortSignal): Promise<string> {
  const form = new FormData();
  form.append("audio", new Blob([wav], { type: "audio/wav" }), "speech.wav");
  const res = await apiFetch("/voice/transcribe", { method: "POST", body: form, signal, headers: { Accept: "application/json" } });
  if (!res.ok) throw new ApiError(res.status, "/voice/transcribe", await res.text());
  return ((await res.json()) as components["schemas"]["Transcript"]).text;
}

/** Record from the microphone and encode mono 16 kHz PCM WAV for the server. */
export async function recordSpeech(signal?: AbortSignal): Promise<ArrayBuffer> {
  if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
    throw new Error("This browser does not support microphone recording.");
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  try {
    if (signal?.aborted) throw new DOMException("Recording was cancelled.", "AbortError");
    const recorder = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    let stopRecording: ReturnType<typeof setTimeout> | undefined;
    let hasStarted = false;
    let rejectStart: (error: Error) => void = () => {};
    let rejectRecording: (error: Error) => void = () => {};
    const started = new Promise<void>((resolve, reject) => {
      rejectStart = reject;
      recorder.onstart = () => { hasStarted = true; resolve(); };
    });
    const recorded = new Promise<Blob>((resolve, reject) => {
      rejectRecording = reject;
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        if (stopRecording) clearTimeout(stopRecording);
        resolve(new Blob(chunks, { type: recorder.mimeType }));
      };
    });
    recorder.onerror = () => {
      const error = new Error("The microphone recording failed.");
      if (hasStarted) rejectRecording(error);
      else rejectStart(error);
    };
    const stop = () => { if (recorder.state !== "inactive") recorder.stop(); };
    signal?.addEventListener("abort", stop, { once: true });
    recorder.start();
    await started;
    stopRecording = setTimeout(stop, 15_000);
    const blob = await recorded;
    signal?.removeEventListener("abort", stop);
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const source = decoded.getChannelData(0);
      const ratio = decoded.sampleRate / 16_000;
      const samples = new Int16Array(Math.floor(source.length / ratio));
      for (let index = 0; index < samples.length; index++) {
        const value = Math.max(-1, Math.min(1, source[Math.floor(index * ratio)]));
        samples[index] = value < 0 ? value * 0x8000 : value * 0x7fff;
      }
      const wav = new ArrayBuffer(44 + samples.byteLength);
      const view = new DataView(wav);
      const write = (offset: number, value: string) => [...value].forEach((char, index) => view.setUint8(offset + index, char.charCodeAt(0)));
      write(0, "RIFF");
      view.setUint32(4, 36 + samples.byteLength, true);
      write(8, "WAVE");
      write(12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, 16_000, true);
      view.setUint32(28, 32_000, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      write(36, "data");
      view.setUint32(40, samples.byteLength, true);
      new Int16Array(wav, 44).set(samples);
      return wav;
    } finally {
      await context.close();
    }
  } finally {
    stream.getTracks().forEach((track) => track.stop());
  }
}

/** The "Speak replies" setting, remembered in this browser. */
export function useSpeakReplies(): [boolean, (on: boolean) => void] {
  const [on, setOn] = useState(false);
  useEffect(() => {
    try {
      setOn(localStorage.getItem(SPEAK_KEY) === "1");
    } catch {
      // storage can be blocked; the setting then lasts for the visit
    }
  }, []);
  const set = useCallback((next: boolean) => {
    setOn(next);
    try {
      localStorage.setItem(SPEAK_KEY, next ? "1" : "0");
    } catch {
      // see above
    }
  }, []);
  return [on, set];
}

export type Speaker = {
  /** Say `text` aloud. Resolves when it has been heard to the end or `stop` cut it off; rejects with a sentence when it cannot be spoken. */
  play: (text: string) => Promise<void>;
  /** Cut off what is playing, or being made. */
  stop: () => void;
  /** Whether audio is being made or played. */
  speaking: boolean;
};

/** Plays text through POST /voice/speak, one reply at a time (a new `play` replaces the last). Stops when the page unmounts. */
export function useSpeaker(): Speaker {
  const [speaking, setSpeaking] = useState(false);
  const current = useRef<{ ctl: AbortController; finish: () => void } | null>(null);

  const stop = useCallback(() => {
    const now = current.current;
    current.current = null;
    now?.ctl.abort();
    now?.finish();
    setSpeaking(false);
  }, []);

  const play = useCallback(
    async (text: string) => {
      stop();
      const ctl = new AbortController();
      let finish = () => {};
      const mine = { ctl, finish: () => finish() };
      current.current = mine;
      setSpeaking(true);
      let url = "";
      let audio: HTMLAudioElement | null = null;
      try {
        const res = await apiFetch("/voice/speak", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "audio/wav" },
          body: JSON.stringify({ text, voice: storedTtsVoice() } satisfies components["schemas"]["SpeakRequest"]),
          signal: ctl.signal,
        });
        if (!res.ok) throw new ApiError(res.status, "/voice/speak", await res.text());
        const blob = await res.blob();
        if (ctl.signal.aborted) return;
        url = URL.createObjectURL(blob);
        const el = new Audio(url);
        audio = el;
        await new Promise<void>((resolve, reject) => {
          finish = resolve;
          el.onended = () => resolve();
          el.onerror = () => reject(new Error("The browser could not play the spoken reply."));
          el.play().catch((error: unknown) =>
            reject(new Error(error instanceof DOMException && error.name === "NotAllowedError" ? "The browser blocked the audio. Click the page once and try again." : "The browser could not play the spoken reply.")),
          );
        });
      } catch (error) {
        if (ctl.signal.aborted) return;
        throw new Error(voiceMessage(error));
      } finally {
        audio?.pause();
        if (url) URL.revokeObjectURL(url);
        if (current.current === mine) {
          current.current = null;
          setSpeaking(false);
        }
      }
    },
    [stop],
  );

  useEffect(() => stop, [stop]);
  return { play, stop, speaking };
}
