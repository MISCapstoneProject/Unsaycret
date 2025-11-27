import React, { createContext, useContext, useState, ReactNode } from 'react';
import { useASRWebStreamBare } from '../hooks/useASRWebStreamBare';
import { API_ENDPOINTS } from '../config/api';
import { SUBTITLE_CONFIG } from '../config/subtitle';
import { speakerColorManager } from '../utils/speakerColors';
import { Session, WebSocketSubtitle } from '../types';

/**
 * 字幕項目介面
 */
interface SubtitleItem {
  id: string;
  speechLogUuid?: string; // 資料庫中對應的 SpeechLog UUID
  content: string;
  timestamp: string;
  isNew: boolean;
  speakerName: string;
  text: string;
  previousText?: string;
  audioPath?: string; // 語者音檔路徑
}

/**
 * 錄音上下文介面
 */
interface RecordingContextType {
  // 狀態
  isRecording: boolean;
  subtitles: SubtitleItem[];
  selectedSession: Session | null;
  
  // 方法
  startRecording: () => void;
  stopRecording: () => void;
  setSelectedSession: (session: Session | null) => void;
  clearSubtitles: () => void;
  updateSubtitle: (id: string, newContent: string) => void;
  deleteSubtitle: (id: string) => void;
  setSubtitleIsNew: (id: string, isNew: boolean) => void;
  playSubtitleAudio: (audioPath: string) => void;  // 播放音檔
}

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

/**
 * RecordingProvider 組件
 */
export const RecordingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [subtitles, setSubtitles] = useState<SubtitleItem[]>([]);
  
  // 🚀 音檔快取：避免重複下載同個音檔
  const audioCache = React.useRef<Map<string, HTMLAudioElement>>(new Map());
  const [isRecording, setIsRecording] = useState<boolean>(false);
  
  const mergeSameSpeaker = SUBTITLE_CONFIG.mergeSameSpeaker;

  // 使用 WebSocket hook
  const { start, stop } = useASRWebStreamBare({
    wsUrl: selectedSession
      ? `${API_ENDPOINTS.stream}?session=${selectedSession.uuid}`
      : API_ENDPOINTS.stream,
    onMessage: (data: any) => {
      // 處理字幕訊息
      if (data.type === "subtitle") {
        const subtitle = data as WebSocketSubtitle;
        const speakerName = subtitle.speakerName || '未知語者';
        
        // 格式化時間
        const formatTime = (dateStr: string) => {
          const date = new Date(dateStr);
          return date.toLocaleTimeString('zh-TW', { 
            hour12: true, 
            hour: 'numeric', 
            minute: '2-digit' 
          });
        };
        
        const timeInfo = subtitle.absoluteStartTime 
          ? formatTime(subtitle.absoluteStartTime)
          : formatTime(new Date().toISOString());
        
        const content = `${speakerName}：${subtitle.text || ''}`;
        
        // 添加到字幕列表
        const newSubtitle: SubtitleItem = {
          id: `${subtitle.segmentId}-${Date.now()}`,
          speechLogUuid: subtitle.speechLogUuid, // 儲存資料庫 UUID 供編輯/刪除使用
          content: content,
          timestamp: timeInfo,
          isNew: true,
          speakerName: speakerName,
          text: subtitle.text || '',
          audioPath: subtitle.audioPath // 儲存音檔路徑
        };
        
        // 🚀 音檔預載優化：字幕一出現就開始預載音檔（背景下載，不阻塞介面）
        if (subtitle.audioPath && !audioCache.current.has(subtitle.audioPath)) {
          const audioUrl = `${API_ENDPOINTS.base}/audio/${subtitle.audioPath}`;
          const audio = new Audio(audioUrl);
          audio.preload = 'auto'; // 自動預載完整音檔
          audioCache.current.set(subtitle.audioPath, audio);
          console.log('🎵 背景預載音檔:', subtitle.audioPath);
        }
        
        setSubtitles((prev) => {
          // 檢查是否開啟合併功能且與上一個字幕是同一語者
          if (mergeSameSpeaker && prev.length > 0) {
            const lastSubtitle = prev[prev.length - 1];
            if (lastSubtitle.speakerName === speakerName) {
              // 同一語者，合併文字
              const updatedPrev = prev.slice(0, -1).map(item => ({ ...item, isNew: false }));
              const mergedSubtitle: SubtitleItem = {
                ...lastSubtitle,
                text: lastSubtitle.text + ' ' + newSubtitle.text,
                content: `${speakerName}：${lastSubtitle.text + ' ' + newSubtitle.text}`,
                timestamp: timeInfo,
                isNew: true,
                previousText: `${speakerName}：${lastSubtitle.text}`
              };
              return [...updatedPrev, mergedSubtitle];
            }
          }
          
          // 不同語者或第一個字幕，創建新的字幕卡片
          const updatedPrev = prev.map(item => ({ ...item, isNew: false }));
          return [...updatedPrev, newSubtitle];
        });
      } else {
        // 其他訊息
        console.log('📩 WebSocket message:', data);
      }
    },
    onError: (err) => {
      console.error('❌ WebSocket error:', err);
    },
    onState: (s) => {
      setIsRecording(s.recording);
    },
  });

  /**
   * 開始錄音
   */
  const startRecording = () => {
    start();
  };

  /**
   * 停止錄音
   */
  const stopRecording = () => {
    stop();
  };

  /**
   * 清除字幕
   */
  const clearSubtitles = () => {
    setSubtitles([]);
    speakerColorManager.reset();
  };

  /**
   * 更新字幕內容（同時更新前端和資料庫）
   */
  const updateSubtitle = async (id: string, newContent: string) => {
    // 先找到對應的字幕項目
    const subtitle = subtitles.find(item => item.id === id);
    
    // 更新前端狀態
    setSubtitles((prev) =>
      prev.map((item) =>
        item.id === id
          ? {
              ...item,
              content: newContent,
              text: newContent.replace(/^[^：]+：/, ''),
              isNew: false
            }
          : item
      )
    );

    // 如果有 speechLogUuid，同步更新資料庫
    if (subtitle?.speechLogUuid) {
      try {
        const { apiService } = await import('../services/api');
        await apiService.updateSpeechLog(subtitle.speechLogUuid, {
          content: newContent.replace(/^[^：]+：/, ''), // 移除語者名稱，只保留純文字
        });
        console.log(`✅ 成功更新資料庫 SpeechLog: ${subtitle.speechLogUuid}`);
      } catch (error) {
        console.error('❌ 更新資料庫失敗:', error);
        // 可以在這裡加入錯誤提示給使用者
      }
    }
  };

  /**
   * 刪除字幕（同時刪除前端和資料庫）
   */
  const deleteSubtitle = async (id: string) => {
    // 先找到對應的字幕項目
    const subtitle = subtitles.find(item => item.id === id);
    
    // 刪除前端狀態
    setSubtitles((prev) => prev.filter((item) => item.id !== id));

    // 如果有 speechLogUuid，同步刪除資料庫
    if (subtitle?.speechLogUuid) {
      try {
        const { apiService } = await import('../services/api');
        await apiService.deleteSpeechLog(subtitle.speechLogUuid);
        console.log(`✅ 成功刪除資料庫 SpeechLog: ${subtitle.speechLogUuid}`);
      } catch (error) {
        console.error('❌ 刪除資料庫失敗:', error);
        // 可以在這裡加入錯誤提示給使用者
      }
    }
  };

  /**
   * 設置字幕的 isNew 狀態
   */
  const setSubtitleIsNew = (id: string, isNew: boolean) => {
    setSubtitles((prev) =>
      prev.map((item) => (item.id === id ? { ...item, isNew } : item))
    );
  };

  /**
   * 播放音檔（優化版：使用快取避免重複下載）
   */
  const playSubtitleAudio = (audioPath: string) => {
    if (!audioPath) {
      console.warn('⚠️ 沒有音檔路徑');
      return;
    }
    
    // 🚀 檢查快取，避免重複下載同個音檔
    let audio = audioCache.current.get(audioPath);
    
    if (!audio) {
      // 第一次播放：創建新的 Audio 物件並快取
      const audioUrl = `${API_ENDPOINTS.base}/audio/${audioPath}`;
      audio = new Audio(audioUrl);
      
      // 預載入音檔元數據（加速首次播放）
      audio.preload = 'metadata';
      
      // 快取這個 Audio 物件
      audioCache.current.set(audioPath, audio);
      
      console.log('🎵 首次載入音檔:', audioPath);
    } else {
      // 使用快取的 Audio 物件，直接播放
      console.log('⚡ 使用快取音檔:', audioPath);
    }
    
    // 重置播放位置並播放
    audio.currentTime = 0;
    audio.play().catch(error => {
      console.error('❌ 播放音檔失敗:', error);
    });
  };

  const value: RecordingContextType = {
    isRecording,
    subtitles,
    selectedSession,
    startRecording,
    stopRecording,
    setSelectedSession,
    clearSubtitles,
    updateSubtitle,
    deleteSubtitle,
    setSubtitleIsNew,
    playSubtitleAudio,
  };

  return (
    <RecordingContext.Provider value={value}>
      {children}
    </RecordingContext.Provider>
  );
};

/**
 * 使用錄音上下文的 Hook
 */
export const useRecording = (): RecordingContextType => {
  const context = useContext(RecordingContext);
  if (!context) {
    throw new Error('useRecording must be used within a RecordingProvider');
  }
  return context;
};
