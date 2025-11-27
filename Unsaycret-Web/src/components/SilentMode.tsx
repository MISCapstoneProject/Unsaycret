import React, { useState, useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { Mic, Plus, Play, Square } from 'lucide-react';
import { Session, Mode } from '../types';
import { apiService } from '../services/api';
import AIPanel from './AIPanel';
import MobileAIDrawer from './MobileAIDrawer';
import { useIsMobile } from '../hooks/useIsMobile';
import SubtitleLine from './SubtitleLine';
import { speakerColorManager } from '../utils/speakerColors';
import { useRecording } from '../contexts/RecordingContext';

const SilentMode: React.FC = () => {
  const isMobile = useIsMobile(1024);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [mode, setMode] = useState<Mode>('stream');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [isLoadingSessions, setIsLoadingSessions] = useState(false);
  const subtitlesEndRef = useRef<HTMLDivElement>(null);
  
  // 使用全局錄音 Context
  const {
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
  } = useRecording();

  const fetchSessions = async () => {
    setIsLoadingSessions(true);
    try {
      const fetchedSessions = await apiService.fetchSessions();
      setSessions(fetchedSessions);
      if (!selectedSession && fetchedSessions.length > 0) {
        setSelectedSession(fetchedSessions[0]);
      }
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setIsLoadingSessions(false);
    }
  };

  // 用 ref 記錄上一次的場合 UUID，避免組件重新渲染時誤清空
  const prevSessionUuidRef = useRef<string | null>(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  // 監聽場合變化，只在使用者手動切換時清除字幕記錄
  useEffect(() => {
    const currentUuid = selectedSession?.uuid || null;
    
    // 如果之前有場合，且現在切換到不同的場合，才清空字幕
    if (prevSessionUuidRef.current !== null && prevSessionUuidRef.current !== currentUuid) {
      clearSubtitles();
      speakerColorManager.reset();
    }
    
    // 更新記錄的場合 UUID
    prevSessionUuidRef.current = currentUuid;
  }, [selectedSession?.uuid]);

  // 自動滾動到最新字幕
  useEffect(() => {
    if (subtitlesEndRef.current) {
      subtitlesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [subtitles]);

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header Section */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
                場合選擇：
              </label>
              <select
                value={selectedSession?.uuid || ''}
                onChange={(e) => {
                  const session = sessions.find(
                    (s) => s.uuid === e.target.value
                  );
                  setSelectedSession(session || null);
                }}
                className="input-field max-w-xs"
                disabled={isLoadingSessions}
              >
                <option value="">選擇場合</option>
                {sessions.map((session) => (
                  <option key={session.uuid} value={session.uuid}>
                    {session.title || session.uuid.slice(0, 8)}
                  </option>
                ))}
              </select>
            </div>

            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-secondary flex items-center space-x-2"
            >
              <Plus className="w-4 h-4" />
              <span>新增</span>
            </button>
          </div>

          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 whitespace-nowrap">
              模式：
            </label>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value as Mode)}
              className="input-field"
              disabled={isRecording}
            >
              <option value="stream">即時串流</option>
              <option value="transcribe">檔案轉錄</option>
            </select>
          </div>
        </div>
      </div>

      {/* Recording Control */}
      <div className="card text-center">
        <button
          onClick={toggleRecording}
          // disabled={isUploading}
          className={`
            inline-flex items-center space-x-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-200 transform hover:scale-105
            ${
              isRecording
                ? 'bg-red-600 hover:bg-red-700 text-white recording-pulse'
                : 'bg-green-600 hover:bg-green-700 text-white'
            }
            
          `}
        >
          {isRecording ? (
            <>
              <Square className="w-6 h-6" />
              <span>停止錄音</span>
            </>
          ) : (
            <>
              <Play className="w-6 h-6" />
              <span>開始錄音</span>
            </>
          )}
        </button>

        {selectedSession && (
            <p className="mt-3 text-sm text-gray-600">
            場合：
            {selectedSession.title || selectedSession.uuid.slice(0, 8)}
            <span className="mx-2"></span>
            類型：
            {selectedSession.session_type || '未設定'}
            </p>
        )}
      </div>

      {/* Output + AI */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
            <Mic className="w-5 h-5" />
            <span>轉錄結果</span>
          </h3>
          <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
            {subtitles.length === 0 ? (
              <p className="text-gray-500 text-center py-8">尚無轉錄結果</p>
            ) : (
              <div className="space-y-3">
                {subtitles.map((subtitle) => (
                  <SubtitleLine
                    key={subtitle.id}
                    content={subtitle.content}
                    timestamp={subtitle.timestamp}
                    isNew={subtitle.isNew}
                    previousContent={subtitle.previousText}
                    speakerName={subtitle.speakerName}
                    audioPath={subtitle.audioPath}
                    onStartEdit={() => {
                      // 進入編輯模式時停止打字機效果
                      setSubtitleIsNew(subtitle.id, false);
                    }}
                    onEdit={(newContent: string) => {
                      // 更新字幕內容
                      updateSubtitle(subtitle.id, newContent);
                    }}
                    onDelete={() => {
                      // 刪除字幕
                      deleteSubtitle(subtitle.id);
                    }}
                    onPlayAudio={(audioPath: string) => {
                      // 播放音檔
                      playSubtitleAudio(audioPath);
                    }}
                  />
                ))}
                <div ref={subtitlesEndRef} />
              </div>
            )}
          </div>
          {subtitles.length > 0 && (
            <button
              onClick={() => {
                clearSubtitles();
                speakerColorManager.reset(); // 重置語者顏色分配
              }}
              className="mt-4 btn-secondary text-sm"
            >
              清除記錄
            </button>
          )}
        </div>
        {!isMobile && (
          <AIPanel outputs={subtitles.map(s => `${s.timestamp} ${s.content}`)} />
        )}
      </div>

      {isMobile && (
        <MobileAIDrawer
          outputs={subtitles.map(s => `${s.timestamp} ${s.content}`)}
          title="AI 助手"
        />
      )}

      {/* Create Session Modal */}
      {showCreateModal && (
        <CreateSessionModal
          onClose={() => setShowCreateModal(false)}
          onSuccess={() => {
            setShowCreateModal(false);
            fetchSessions();
          }}
        />
      )}
    </div>
  );
};

interface CreateSessionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const CreateSessionModal: React.FC<CreateSessionModalProps> = ({
  onClose,
  onSuccess,
}) => {
  const [title, setTitle] = useState('');
  const [sessionType, setSessionType] = useState('');
  const [participants, setParticipants] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim()) return;

    setIsCreating(true);
    try {
      const participantList = participants
        .split(',')
        .map((p) => p.trim())
        .filter((p) => p.length > 0);

      await apiService.createSession({
        title: title.trim(),
        session_type: sessionType.trim() || '',
        participants: participantList.length > 0 ? participantList : [],
      });

      onSuccess();
    } catch (error) {
      console.error('Failed to create session:', error);
    } finally {
      setIsCreating(false);
    }
  };

  return createPortal(
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4">
        <h2 className="text-xl font-bold text-gray-900 mb-4">新增場合</h2>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              名稱 *
            </label>
            <input
              type="text"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="input-field"
              placeholder="輸入場合名稱"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              類型 *
            </label>
            <input
              type="text"
              value={sessionType}
              onChange={(e) => setSessionType(e.target.value)}
              className="input-field"
              placeholder="例：會議、訪談、討論"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              參與者UUID（用逗號分隔）
            </label>
            <input
              type="text"
              value={participants}
              onChange={(e) => setParticipants(e.target.value)}
              className="input-field"
              placeholder="uuid1, uuid2, uuid3"
            />
          </div>

          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary flex-1"
              disabled={isCreating}
            >
              取消
            </button>
            <button
              type="submit"
              className="btn-primary flex-1"
              disabled={isCreating || !title.trim()}
            >
              {isCreating ? '建立中...' : '建立'}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
};

export default SilentMode;
