export interface Speaker {
  uuid: string;
  speaker_id?: number;
  full_name?: string;
  nickname?: string;
  gender?: string;
  created_at?: string;
  last_active_at?: string;
  meet_count?: number;
  meet_days?: number;
  voiceprint_ids: string[];
  first_audio?: string;
}

export interface ParticipantDetail {
  uuid: string;
  full_name?: string;
  nickname?: string;
}

export interface Session {
  uuid: string;
  session_id?: string;
  session_type?: string;
  title?: string;
  start_time?: string;
  end_time?: string;
  summary?: string;
  participants?: string[];  // UUID 列表 (向後兼容)
  participants_details?: ParticipantDetail[];  // 完整資訊
}

export interface SpeechLog {
  uuid: string;
  content?: string;
  timestamp?: string;
  confidence?: number;
  duration?: number;
  language?: string;
  speaker?: string;  // Speaker UUID (保留相容性)
  // ✅ 新增欄位: Speaker 的名字和暱稱
  speaker_name?: string;      // Speaker 的全名
  speaker_nickname?: string;  // Speaker 的暱稱
  session?: string;
}

export type Mode = 'transcribe' | 'stream';

export interface RecordingState {
  isRecording: boolean;
  isUploading: boolean;
  mode: Mode;
  sessionID?: string;
  outputs: string[];
}

// WebSocket v2.0 協議類型定義
export interface WebSocketSubtitle {
  type: "subtitle";
  segmentId: string;
  speakerId: string;
  speakerName: string;
  speechLogUuid?: string; // SpeechLog UUID (用於前端編輯/刪除)
  distance: number;
  text: string;
  confidence: number;
  startTime: number;
  endTime: number;
  absoluteStartTime: string;
  absoluteEndTime: string;
  audioPath?: string; // 語者音檔路徑
  isFinal: boolean;
  segment: {
    totalSpeakers: number;
    speakerIndex: number;
    segmentStart: number;
    segmentEnd: number;
  };
}

export interface WebSocketAudioMessage {
  type: "audio";
  timestamp: number;
  data: string | ArrayBuffer;
}

export interface WebSocketStopMessage {
  type: "stop";
}

export type WebSocketMessage = WebSocketSubtitle | WebSocketAudioMessage | WebSocketStopMessage;