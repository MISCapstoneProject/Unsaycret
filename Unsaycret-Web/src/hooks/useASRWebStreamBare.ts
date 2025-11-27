// 僅做 Float32 -> Int16 PCM 的最小處理版本（不重採樣、不正規化、不增益）

import { useRef, useCallback } from 'react';
import { WebSocketMessage } from '../types';

type ASRWebStreamBareOptions = {
  wsUrl: string;
  onMessage?: (data: WebSocketMessage | any) => void;
  onError?: (err: string) => void;
  onState?: (s: { recording: boolean; connected: boolean }) => void;
};

export function useASRWebStreamBare(opts: ASRWebStreamBareOptions) {
  const { wsUrl, onMessage, onError, onState } = opts;

  // 使用 useRef 保持這些變數在組件生命週期中的持久性
  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const procNodeRef = useRef<ScriptProcessorNode | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const recordingRef = useRef(false);
  const connectedRef = useRef(false);

  const notifyState = useCallback(() => {
    onState?.({ recording: recordingRef.current, connected: connectedRef.current });
  }, [onState]);

  const start = useCallback(async () => {
    try {
      if (recordingRef.current) return;

      // 建立 WebSocket 連線
      wsRef.current = new WebSocket(wsUrl);
      wsRef.current.binaryType = 'arraybuffer';
      wsRef.current.onopen = () => {
        connectedRef.current = true;
        notifyState();
      };
      wsRef.current.onclose = () => {
        connectedRef.current = false;
        notifyState();
      };
      wsRef.current.onerror = () => onError?.('WebSocket error');
      wsRef.current.onmessage = (e) => {
        try {
          const data = typeof e.data === 'string' ? JSON.parse(e.data) : e.data;
          onMessage?.(data);
        } catch {}
      };

      // 取得麥克風
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
        video: false,
      });

      // 建立 AudioContext 與 ScriptProcessor
      audioCtxRef.current = new AudioContext();
      const source = audioCtxRef.current.createMediaStreamSource(mediaStreamRef.current);
      sourceNodeRef.current = source;
      procNodeRef.current = audioCtxRef.current.createScriptProcessor(4096, 1, 1);

      procNodeRef.current.onaudioprocess = (evt) => {
        if (!recordingRef.current) return;
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        
        const input = evt.inputBuffer.getChannelData(0);
        const f32 = new Float32Array(input);
        wsRef.current.send(f32.buffer);
      };

      source.connect(procNodeRef.current);
      procNodeRef.current.connect(audioCtxRef.current.destination);

      recordingRef.current = true;
      notifyState();
    } catch (err: any) {
      onError?.(err?.message ?? String(err));
      await stop();
    }
  }, [wsUrl, onMessage, onError, notifyState]);

  const stop = useCallback(async () => {
    try {
      recordingRef.current = false;

      // 清理音訊處理節點
      if (procNodeRef.current) {
        try { procNodeRef.current.disconnect(); } catch {}
        procNodeRef.current.onaudioprocess = null;
        procNodeRef.current = null;
      }
      if (sourceNodeRef.current) {
        try { sourceNodeRef.current.disconnect(); } catch {}
        sourceNodeRef.current = null;
      }
      if (audioCtxRef.current) {
        try { await audioCtxRef.current.close(); } catch {}
        audioCtxRef.current = null;
      }
      if (mediaStreamRef.current) {
        for (const t of mediaStreamRef.current.getTracks()) t.stop();
        mediaStreamRef.current = null;
      }

      // 發送停止信號並關閉 WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        try {
          console.log("🛑 發送停止信號");
          wsRef.current.send("stop");
          await new Promise(resolve => setTimeout(resolve, 100));
        } catch (e) {
          console.error("發送停止信號失敗:", e);
        }
        try {
          wsRef.current.close();
        } catch {}
        wsRef.current = null;
        connectedRef.current = false;
      }

      notifyState();
    } catch (err: any) {
      onError?.(err?.message ?? String(err));
    }
  }, [onError, notifyState]);

  return { start, stop };
}




