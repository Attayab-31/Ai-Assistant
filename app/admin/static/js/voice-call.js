/**
 * Shared voice-call mic streaming + agent audio playback for test_console and
 * client_call. Both pages talk to the same /test/api WebSocket protocol.
 */
(function (global) {
  "use strict";

  const SILENT_WAV =
    "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA==";

  function unlockAudioPlayback(playerId, volume) {
    const player = document.getElementById(playerId || "audioPlayer");
    if (!player || player.dataset.audioUnlocked === "1") return;
    player.src = SILENT_WAV;
    player.volume = 0.001;
    const p = player.play();
    if (p && typeof p.then === "function") {
      p.then(() => {
        player.dataset.audioUnlocked = "1";
        try {
          player.pause();
          player.currentTime = 0;
        } catch (e) {}
        player.removeAttribute("src");
        try {
          player.load();
        } catch (e) {}
        player.volume = volume ?? 0.85;
      }).catch(() => {});
    }
  }

  class VoiceStreamingMic {
    constructor(ws) {
      this.ws = ws;
      this.stream = null;
      this.ctx = null;
      this.processor = null;
      this.active = false;
      this.muted = true;
      this.agentSpeaking = false;
    }

    async start() {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.ctx.createMediaStreamSource(this.stream);
      const bufferSize = 2048;
      this.processor = this.ctx.createScriptProcessor(bufferSize, 1, 1);
      this.processor.onaudioprocess = (e) => this._onAudio(e);
      source.connect(this.processor);
      const silent = this.ctx.createGain();
      silent.gain.value = 0;
      this.processor.connect(silent);
      silent.connect(this.ctx.destination);
      this.active = true;
    }

    _floatToPcm16Chunk(input) {
      const ratio = this.ctx.sampleRate / 8000;
      const outLen = Math.max(1, Math.floor(input.length / ratio));
      const int16 = new Int16Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const idx = Math.min(input.length - 1, Math.floor(i * ratio));
        int16[i] = Math.max(-32768, Math.min(32767, Math.round(input[idx] * 32767)));
      }
      return int16;
    }

    _sendMediaChunk(int16) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const bytes = new Uint8Array(int16.buffer);
      let bin = "";
      for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      this.ws.send(
        JSON.stringify({
          event: "media",
          media: {
            payload: btoa(bin),
            encoding: "linear16",
            sample_rate: 8000,
          },
        })
      );
    }

    _onAudio(e) {
      if (!this.active) return;
      const input = e.inputBuffer.getChannelData(0);
      if (this.agentSpeaking) {
        this._sendMediaChunk(this._floatToPcm16Chunk(input));
        return;
      }
      if (this.muted) return;
      this._sendMediaChunk(this._floatToPcm16Chunk(input));
    }

    unmute() {
      this.muted = false;
      this.agentSpeaking = false;
    }

    mute() {
      this.muted = true;
      this.agentSpeaking = false;
    }

    stop() {
      this.active = false;
      this.mute();
      if (this.processor) {
        this.processor.disconnect();
        this.processor = null;
      }
      if (this.ctx) {
        this.ctx.close().catch(() => {});
        this.ctx = null;
      }
      if (this.stream) {
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
      }
    }
  }

  function b64ToBytes(b64) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return arr;
  }

  /**
   * Agent audio queue + playback for one <audio> element.
   * opts: { playerId, volume, postPlaybackMuteMs, getMic, onSpeaking, onMicLive, onStopPlayback }
   */
  function createPlaybackManager(opts) {
    const playerId = opts.playerId || "audioPlayer";
    const volume = opts.volume ?? 0.85;
    const postPlaybackMuteMs = opts.postPlaybackMuteMs ?? 350;
    const getMic = opts.getMic || (() => null);
    const onSpeaking = opts.onSpeaking || (() => {});
    const onMicLive = opts.onMicLive || (() => {});
    const onStopPlayback = opts.onStopPlayback || (() => {});

    const agentAudioQueue = [];
    let agentAudioPlaying = false;
    let playSessionId = 0;
    let activeBlobUrl = null;
    let playbackWatchdog = null;
    let pendingUnmute = false;
    let agentQuietUntil = 0;

    function $(id) {
      return document.getElementById(id);
    }

    function isPlayerBusy() {
      const player = $(playerId);
      return !!(player && player.src && !player.paused && !player.ended);
    }

    function invalidatePlayback() {
      playSessionId += 1;
    }

    function resetAudioElement() {
      const player = $(playerId);
      if (!player) return;
      player.onended = null;
      player.onerror = null;
      try {
        player.pause();
        player.currentTime = 0;
      } catch (e) {}
      if (activeBlobUrl) {
        URL.revokeObjectURL(activeBlobUrl);
        activeBlobUrl = null;
      }
      player.removeAttribute("src");
      try {
        player.load();
      } catch (e) {}
    }

    function playAudio(b64, onEnded) {
      if (!b64) {
        if (onEnded) onEnded();
        return;
      }
      const session = playSessionId;
      const player = $(playerId);
      if (!player) {
        if (onEnded) onEnded();
        return;
      }
      const finish = () => {
        if (session !== playSessionId) return;
        if (onEnded) onEnded();
      };
      player.onended = finish;
      player.onerror = () => finish();
      if (activeBlobUrl) {
        URL.revokeObjectURL(activeBlobUrl);
        activeBlobUrl = null;
      }
      try {
        const blob = new Blob([b64ToBytes(b64)], { type: "audio/wav" });
        activeBlobUrl = URL.createObjectURL(blob);
        player.src = activeBlobUrl;
      } catch (e) {
        player.src = "data:audio/wav;base64," + b64;
      }
      player.volume = volume;
      player.play().catch(() => finish());
    }

    function onAgentSpeaking() {
      pendingUnmute = false;
      const mic = getMic();
      if (mic) mic.agentSpeaking = true;
      onSpeaking();
    }

    function tryUnmuteMic() {
      if (!pendingUnmute) return;
      const mic = getMic();
      if (!mic) return;
      const wait = agentQuietUntil - Date.now();
      if (wait > 0) {
        setTimeout(tryUnmuteMic, wait);
        return;
      }
      pendingUnmute = false;
      mic.unmute();
      onMicLive();
    }

    function onAgentDone() {
      const mic = getMic();
      if (mic) mic.agentSpeaking = false;
      pendingUnmute = true;
      agentQuietUntil = Date.now() + postPlaybackMuteMs;
      tryUnmuteMic();
    }

    function playNextAgentAudio() {
      if (!agentAudioQueue.length) {
        agentAudioPlaying = false;
        return;
      }
      agentAudioPlaying = true;
      onAgentSpeaking();
      const { b64, turnEnd } = agentAudioQueue.shift();
      const session = playSessionId;
      playAudio(b64, () => {
        if (session !== playSessionId) return;
        if (turnEnd) onAgentDone();
        playNextAgentAudio();
      });
    }

    function ensureAgentPlayback() {
      if (!agentAudioQueue.length) {
        agentAudioPlaying = false;
        return;
      }
      if (agentAudioPlaying && isPlayerBusy()) return;
      if (agentAudioPlaying && !isPlayerBusy()) {
        agentAudioPlaying = false;
      }
      playNextAgentAudio();
    }

    function schedulePlaybackWatchdog() {
      if (playbackWatchdog) clearTimeout(playbackWatchdog);
      playbackWatchdog = setTimeout(() => {
        playbackWatchdog = null;
        if (!agentAudioQueue.length) return;
        if (!isPlayerBusy()) {
          agentAudioPlaying = false;
          playNextAgentAudio();
        }
      }, 2000);
    }

    function enqueueAgentAudio(b64, turnEnd = true) {
      agentAudioQueue.push({ b64, turnEnd: turnEnd !== false });
      ensureAgentPlayback();
      schedulePlaybackWatchdog();
    }

    function stopAgentPlayback() {
      invalidatePlayback();
      resetAudioElement();
      agentAudioQueue.length = 0;
      agentAudioPlaying = false;
      pendingUnmute = false;
      if (playbackWatchdog) {
        clearTimeout(playbackWatchdog);
        playbackWatchdog = null;
      }
      const mic = getMic();
      if (mic) {
        mic.agentSpeaking = false;
        mic.muted = false;
      }
      onStopPlayback();
    }

    return {
      enqueueAgentAudio,
      stopAgentPlayback,
      invalidatePlayback,
      resetAudioElement,
      playAudio,
      clearPendingUnmute: () => {
        pendingUnmute = false;
      },
    };
  }

  global.VoiceCall = {
    SILENT_WAV,
    VoiceStreamingMic,
    unlockAudioPlayback,
    createPlaybackManager,
    b64ToBytes,
  };
})(window);
