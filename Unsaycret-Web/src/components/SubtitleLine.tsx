import React, { useState, useEffect, useRef } from 'react';
import IncrementalTypewriterText from './IncrementalTypewriterText';
import { SUBTITLE_CONFIG } from '../config/subtitle';
import { speakerColorManager } from '../utils/speakerColors';
import { Trash2, Volume2 } from 'lucide-react';

interface SubtitleLineProps {
  content: string;
  timestamp?: string;
  isNew?: boolean;
  previousContent?: string;
  speakerName?: string;
  audioPath?: string;  // 音檔路徑
  onEdit?: (newContent: string) => void;
  onDelete?: () => void;
  onStartEdit?: () => void;
  onPlayAudio?: (audioPath: string) => void;  // 播放音檔回調
}

const SubtitleLine: React.FC<SubtitleLineProps> = ({
  content,
  timestamp,
  isNew = false,
  previousContent,
  speakerName,
  audioPath,
  onEdit,
  onDelete,
  onPlayAudio,
  onStartEdit
}) => {
  const [isEditing, setIsEditing] = useState<boolean>(false);
  const [showEditContent, setShowEditContent] = useState<boolean>(false);
  const [editedText, setEditedText] = useState<string>('');
  const cardRef = useRef<HTMLDivElement>(null);
  
  // 獲取語者顏色
  const borderColor = speakerName ? speakerColorManager.getColor(speakerName) : '#f3f4f6';
  
  // 解析 content，分離語者名稱和文字內容
  const parseContent = (fullContent: string): { speaker: string; text: string } => {
    const match = fullContent.match(/^([^：]+)：(.*)$/);
    if (match) {
      return { speaker: match[1], text: match[2] };
    }
    return { speaker: speakerName || '', text: fullContent };
  };
  
  // 當 content 改變時，同步更新 editedText（處理即時更新的情況）
  useEffect(() => {
    if (!isEditing) {
      const { text } = parseContent(content);
      setEditedText(text);
    }
  }, [content, isEditing]);
  
  // 點擊外部自動取消編輯
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (cardRef.current && !cardRef.current.contains(event.target as Node)) {
        if (isEditing) {
          handleCancelEdit();
        }
      }
    };
    
    if (isEditing) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('touchstart', handleClickOutside as any);
    }
    
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside as any);
    };
  }, [isEditing]);
  
  // 控制編輯內容的顯示（延遲顯示/隱藏）
  useEffect(() => {
    if (isEditing) {
      // 進入編輯模式：立即顯示編輯內容
      setShowEditContent(true);
    } else {
      // 退出編輯模式：延遲 80ms 後隱藏編輯內容（讓按鈕先消失）
      const timer = setTimeout(() => {
        setShowEditContent(false);
      }, 90);
      return () => clearTimeout(timer);
    }
  }, [isEditing]);
  
  /**
   * 進入編輯模式
   */
  const handleStartEdit = () => {
    const { text } = parseContent(content);
    setIsEditing(true);
    setEditedText(text);
    // 通知父組件停止打字機效果
    if (onStartEdit) {
      onStartEdit();
    }
  };
  
  /**
   * 確認編輯
   */
  const handleConfirmEdit = () => {
    if (onEdit && editedText.trim() !== '') {
      const { speaker } = parseContent(content);
      const newContent = `${speaker}：${editedText.trim()}`;
      onEdit(newContent);
    }
    setIsEditing(false);
  };
  
  /**
   * 取消編輯
   */
  const handleCancelEdit = () => {
    const { text } = parseContent(content);
    setEditedText(text);
    setIsEditing(false);
  };
  
  /**
   * 處理刪除
   */
  const handleDelete = () => {
    if (onDelete && window.confirm('確定要刪除這條字幕嗎？')) {
      onDelete();
    }
  };
  
  /**
   * 處理按鍵事件（Enter 確認，Escape 取消）
   */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleConfirmEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };
  
  return (
    <div 
      ref={cardRef}
      className={`
        subtitle-line bg-white rounded-lg shadow-sm border-2 
        hover:shadow-md transition-all duration-300 overflow-hidden
        ${isEditing ? 'ring-2 ring-primary-300' : 'cursor-pointer'}
      `}
      style={{ borderColor }}
      onClick={!isEditing ? handleStartEdit : undefined}
    >
      {/* 主要內容區域 */}
      <div className="p-4">
        <div className="flex justify-between items-start gap-4">
          <div className="flex items-start gap-3 flex-1">
            {/* 語者顏色圓點 */}
            {speakerName && (
              <div 
                className="w-3 h-3 rounded-full flex-shrink-0 mt-1.5"
                style={{ backgroundColor: borderColor }}
                title={`語者: ${speakerName}`}
              />
            )}
            
            {/* 內容區域 */}
            <div className="flex-1">
              {showEditContent ? (
                // 編輯模式：顯示語者名稱 + 可編輯文字框
                <div className="flex items-start gap-2">
                  <span className="text-sm text-gray-800 whitespace-nowrap mt-0.5">
                    {parseContent(content).speaker}：
                  </span>
                  <textarea
                    value={editedText}
                    onChange={(e) => setEditedText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onClick={(e) => e.stopPropagation()}
                    className="flex-1 text-sm text-gray-800 leading-relaxed border border-gray-300 rounded-md p-2 focus:outline-none focus:ring-2 focus:ring-primary-400 resize-none"
                    rows={Math.max(2, editedText.split('\n').length)}
                    autoFocus
                    placeholder="輸入字幕內容..."
                  />
                </div>
              ) : (
                // 顯示模式
                <div className="text-sm text-gray-800 leading-relaxed min-h-[20px]">
                  {isNew ? (
                    <IncrementalTypewriterText 
                      fullContent={content}
                      previousContent={previousContent}
                      speed={SUBTITLE_CONFIG.typewriter.speed}
                      pauseAtComma={SUBTITLE_CONFIG.typewriter.pauseAtComma}
                    />
                  ) : (
                    <span>{content}</span>
                  )}
                </div>
              )}
            </div>
          </div>
          
          {/* 時間戳記 */}
          {timestamp && (
            <div className="text-xs text-gray-400 font-mono flex-shrink-0 ml-4 mt-0.5 opacity-70">
              {timestamp}
            </div>
          )}
        </div>
      </div>
      
      {/* 按鈕區域（展開動畫） */}
      <div 
        className={`
          transition-all ease-in-out overflow-hidden
          ${isEditing ? 'max-h-16' : 'max-h-0'}
        `}
      >
        <div 
          className={`
            px-4 pb-4 flex items-center justify-between
            transition-opacity duration-200
            ${isEditing ? 'opacity-100 delay-0' : 'opacity-0 delay-0'}
          `}
        >
          {/* 左下：播放音檔 + 刪除按鈕 */}
          <div className="flex items-center gap-2">
            {/* 播放音檔按鈕 */}
            {audioPath && onPlayAudio && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onPlayAudio(audioPath);
                }}
                className="flex items-center gap-1.5 px-2 py-1.5 text-gray-400 hover:text-blue-500 hover:bg-blue-50 rounded-md transition-colors focus:outline-none active:outline-none"
                style={{ outline: 'none', WebkitTapHighlightColor: 'transparent' }}
                title="播放音檔"
              >
                <Volume2 className="w-4 h-4" />
                <span className="text-xs">播放</span>
              </button>
            )}
            
            {/* 刪除按鈕 */}
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete();
              }}
              className="flex items-center gap-1.5 px-2 py-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors focus:outline-none active:outline-none"
              style={{ outline: 'none', WebkitTapHighlightColor: 'transparent' }}
              title="刪除字幕"
            >
              <Trash2 className="w-4 h-4" />
              <span className="text-xs">刪除</span>
            </button>
          </div>
          
          {/* 右下：取消 + 完成按鈕 */}
          <div className="flex items-center gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleCancelEdit();
              }}
              className="px-4 py-1.5 bg-gray-100 text-gray-600 text-sm rounded-md hover:bg-gray-200 transition-colors focus:outline-none active:outline-none"
              style={{ outline: 'none', WebkitTapHighlightColor: 'transparent' }}
            >
              取消
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleConfirmEdit();
              }}
              className="px-4 py-1.5 text-gray-600 text-sm rounded-md hover:opacity-90 transition-all focus:outline-none active:outline-none"
              style={{ 
                backgroundColor: borderColor,
                outline: 'none', 
                WebkitTapHighlightColor: 'transparent'
              }}
            >
              編輯完成
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubtitleLine;