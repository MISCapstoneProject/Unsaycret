// 字幕顯示配置
export const SUBTITLE_CONFIG = {
  // 是否合併同一語者的連續字幕
  mergeSameSpeaker: false,
  
  // 打字機效果設定
  typewriter: {
    speed: 50,           // 打字速度（毫秒）
    pauseAtComma: 300,   // 遇到標點符號的停頓時間（毫秒）
  },
  
  // 時間顯示格式
  timeFormat: {
    hour12: true,        // 使用12小時制
    showSeconds: false,  // 顯示秒數
  },
  
  // 語者顏色配置
  speakerColors: [
    '#e0f2fe', // 淺藍色
    '#f3e5f5', // 淺紫色
    '#e8f5e8', // 淺綠色
    '#fff3e0', // 淺橙色
    '#fce4ec', // 淺粉色
    '#f1f8e9', // 淺青綠色
    '#fff8e1', // 淺黃色
    '#e3f2fd', // 淺天藍色
    '#f9fbe7', // 淺青色
    '#fdf2f8', // 淺玫瑰色
  ],

  // 其他顯示選項
  display: {
    autoScroll: true,         // 自動滾動到最新字幕
    maxSubtitles: 100,        // 最大保留字幕數量（避免記憶體過多使用）
    showConfidence: false,    // 是否顯示信心度
    showSpeakerIndex: false,  // 是否顯示語者編號 [1/2] [2/2]（已停用）
  }
};

export default SUBTITLE_CONFIG;