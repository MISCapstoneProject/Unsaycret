// AI 設定（預留 API Key 供使用者填入）
export const AI_CONFIG = {
  // 從環境變數讀取 OpenAI API Key
  OPENAI_API_KEY: import.meta.env.VITE_OPENAI_API_KEY || '',
  // 從環境變數讀取模型名稱，預設為 gpt-4o-mini
  OPENAI_MODEL: import.meta.env.VITE_OPENAI_MODEL || 'gpt-4o-mini',
};


