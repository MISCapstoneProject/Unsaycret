const BASE_URL = "http://localhost:8000";

export const API_ENDPOINTS = {
  base: BASE_URL,  // 新增 base URL 供音檔路徑使用
  speakers: `${BASE_URL}/speakers`,
  transcribe: `${BASE_URL}/transcribe`,
  stream: BASE_URL
    .replace("https://", "wss://")
    .replace("http://", "ws://") + "/ws/stream",
  sessions: `${BASE_URL}/sessions`,
  speechLogs: `${BASE_URL}/speechlogs`,
  
  speaker: (uuid: string) => `${API_ENDPOINTS.speakers}/${uuid}`,
  speakerSessions: (uuid: string) => `${API_ENDPOINTS.speakers}/${uuid}/sessions`,
  speakerSpeechLogs: (uuid: string) => `${API_ENDPOINTS.speakers}/${uuid}/speechlogs`,
  session: (uuid: string) => `${API_ENDPOINTS.sessions}/${uuid}`,
  sessionSpeechLogs: (uuid: string) => `${API_ENDPOINTS.sessions}/${uuid}/speechlogs`,
  speechLog: (uuid: string) => `${API_ENDPOINTS.speechLogs}/${uuid}`,
};