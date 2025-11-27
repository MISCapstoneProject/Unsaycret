import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Calendar, Plus, Edit3, Trash2, Users, Clock, MessageSquare, Brain, Sparkles, Mic } from 'lucide-react';
import { Session, SpeechLog } from '../types';
import { apiService } from '../services/api';
import ReactMarkdown from 'react-markdown';
import { aiService } from '../services/ai';
import { useRecording } from '../contexts/RecordingContext';

const SessionsMode: React.FC = () => {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { isRecording } = useRecording();

  const fetchSessions = async () => {
    setIsLoading(true);
    try {
      const fetchedSessions = await apiService.fetchSessions();
      setSessions(fetchedSessions);
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const handleDeleteSession = async (session: Session) => {
    if (!confirm(`確定要刪除紀錄「${session.title || session.uuid.slice(0, 8)}」嗎？`)) {
      return;
    }

    try {
      await apiService.deleteSession(session.uuid);
      await fetchSessions();
      if (selectedSession?.uuid === session.uuid) {
        setSelectedSession(null);
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* 錄音中指示器 */}
      {isRecording && (
        <div className="card bg-red-50 border-red-200">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Mic className="w-5 h-5 text-red-600" />
              <div className="absolute -top-1 -right-1 w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
            </div>
            <div>
              <p className="font-medium text-red-900">正在錄音中</p>
              <p className="text-xs text-red-600">背景持續接收即時字幕</p>
            </div>
          </div>
        </div>
      )}
      
      {/* Header */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
            <Calendar className="w-6 h-6" />
            <span>紀錄管理</span>
          </h2>
          
          <div className="flex space-x-3">
            <button
              onClick={fetchSessions}
              className="btn-secondary transform hover:scale-105 active:scale-95 transition-transform duration-150"
              disabled={isLoading}
            >
              {isLoading ? '載入中...' : '重新載入'}
            </button>
            <button
              onClick={() => setShowCreateModal(true)}
              className="btn-primary flex items-center space-x-2 transform hover:scale-105 active:scale-95 transition-transform duration-150"
            >
              <Plus className="w-4 h-4" />
              <span>新增</span>
            </button>
          </div>
        </div>
      </div>

      {/* Sessions Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sessions.map(session => (
            <SessionCard
              key={session.uuid}
              session={session}
              onClick={() => setSelectedSession(session)}
              onDelete={() => handleDeleteSession(session)}
            />
          ))}
        </div>
      )}

      {sessions.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <Calendar className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500 mb-4">尚無任何記錄</p>
          <button
            onClick={() => setShowCreateModal(true)}
            className="btn-primary transform hover:scale-105 active:scale-95 transition-transform duration-150"
          >
            建立
          </button>
        </div>
      )}

      {/* Session Detail Modal */}
      {selectedSession && (
        <SessionDetailModal
          session={selectedSession}
          onClose={() => setSelectedSession(null)}
          onUpdate={fetchSessions}
          onDelete={() => handleDeleteSession(selectedSession)}
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

interface SessionCardProps {
  session: Session;
  onClick: () => void;
  onDelete: () => void;
}

const SessionCard: React.FC<SessionCardProps> = ({ session, onClick, onDelete }) => {
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete();
  };

  return (
    <div
      onClick={onClick}
      className="card cursor-pointer hover:shadow-lg transform hover:-translate-y-1 hover:scale-[1.02] transition-all duration-200 active:scale-[0.98]"
    >
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-900 truncate">
          {session.title || `場合 ${session.uuid.slice(0, 8)}`}
        </h3>
        <button
          onClick={handleDelete}
          className="text-gray-400 hover:text-red-600 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
      
      <div className="space-y-2 text-sm text-gray-600">
        {session.session_type && (
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 bg-primary-500 rounded-full"></span>
            <span>類型：{session.session_type}</span>
          </div>
        )}
        
        {session.participants && (
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4" />
            <span>參與者：{session.participants.length} 人</span>
          </div>
        )}
        
        {session.start_time && (
          <div className="flex items-center space-x-2">
            <Clock className="w-4 h-4" />
            <span>{new Date(session.start_time).toLocaleString('zh-TW')}</span>
          </div>
        )}
      </div>
      
      {session.summary && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500 line-clamp-2">{session.summary}</p>
        </div>
      )}
    </div>
  );
};

interface SessionDetailModalProps {
  session: Session;
  onClose: () => void;
  onUpdate: () => void;
  onDelete: () => void;
}

const SessionDetailModal: React.FC<SessionDetailModalProps> = ({ 
  session, 
  onClose, 
  onUpdate, 
  onDelete 
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({
    title: session.title || '',
    session_type: session.session_type || '',
    summary: session.summary || '',
  });
  const [speechLogs, setSpeechLogs] = useState<SpeechLog[]>([]);
  const [activeTab, setActiveTab] = useState<'info' | 'logs'>('info');
  const [isUpdating, setIsUpdating] = useState(false);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  // ✅ 移除 speakerNames state - 不再需要前端對照表
  
  // 參與者名稱對照 (用於顯示參與者列表)
  const [participantNames, setParticipantNames] = useState<Record<string, string>>({});
  
  // AI 功能狀態
  const [aiSummary, setAiSummary] = useState<string>('');
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [emotionAnalysis, setEmotionAnalysis] = useState<string>('');
  const [isAnalyzingEmotion, setIsAnalyzingEmotion] = useState(false);
  const [question, setQuestion] = useState<string>('');
  const [answer, setAnswer] = useState<string>('');
  const [isAsking, setIsAsking] = useState(false);

  // 動畫控制
  useEffect(() => {
    // 組件掛載後立即顯示動畫
    const timer = setTimeout(() => setIsVisible(true), 10);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const fetchSpeechLogs = async () => {
      setIsLoadingLogs(true);
      try {
        const logs = await apiService.fetchSessionSpeechLogs(session.uuid);
        setSpeechLogs(logs);
        
        // ✅ 後端已經包含 speaker_name 和 speaker_nickname,不需要前端查詢
        // 移除對照表邏輯
        
        // 自動執行 AI 摘要
        if (logs.length > 0) {
          handleAutoSummarize(logs);
        }
      } catch (error) {
        console.error('Failed to fetch speech logs:', error);
      } finally {
        setIsLoadingLogs(false);
      }
    };

    fetchSpeechLogs();
  }, [session.uuid]);
  
  // ✅ 直接從 session.participants_details 取得語者名稱 (不需要額外查詢)
  useEffect(() => {
    if (!session.participants_details || session.participants_details.length === 0) {
      setParticipantNames({});
      return;
    }
    
    const names: Record<string, string> = {};
    session.participants_details.forEach(participant => {
      names[participant.uuid] = participant.nickname || participant.full_name || participant.uuid.slice(0, 8);
    });
    setParticipantNames(names);
  }, [session.participants_details]);
  
  // 自動 AI 摘要
  const handleAutoSummarize = async (logs: SpeechLog[]) => {
    setIsSummarizing(true);
    try {
      const transcript = logs
        .map(l => {
          // ✅ 使用後端回傳的 speaker_nickname 或 speaker_name
          const speakerName = l.speaker_nickname || l.speaker_name || l.speaker?.slice(0, 8) || '';
          return `${speakerName ? speakerName + '：' : ''}${l.content ?? ''}`;
        })
        .filter(Boolean)
        .join('\n');
      
      if (!transcript.trim()) {
        setAiSummary('目前沒有可摘要的內容');
        return;
      }
      
      const result = await aiService.summarize(transcript, { language: 'zh-TW', style: 'bullet' });
      setAiSummary(result);
    } catch (error: any) {
      console.error('AI 摘要失敗:', error);
      setAiSummary(`⚠️ AI 摘要失敗: ${error?.message || '未知錯誤'}`);
    } finally {
      setIsSummarizing(false);
    }
  };
  
  // 手動情緒與互動分析
  const handleEmotionAnalysis = async () => {
    if (speechLogs.length === 0) {
      setEmotionAnalysis('目前沒有可分析的對話內容');
      return;
    }
    
    setIsAnalyzingEmotion(true);
    setEmotionAnalysis('');
    try {
      const transcript = speechLogs
        .map(l => {
          // ✅ 使用後端回傳的 speaker_nickname 或 speaker_name
          const speakerName = l.speaker_nickname || l.speaker_name || l.speaker?.slice(0, 8) || '';
          return `${speakerName ? speakerName + '：' : ''}${l.content ?? ''}`;
        })
        .filter(Boolean)
        .join('\n');
      
      const result = await aiService.analyzeEmotion(transcript, 'zh-TW');
      setEmotionAnalysis(result);
    } catch (error: any) {
      console.error('情緒與互動分析失敗:', error);
      setEmotionAnalysis(`⚠️ 情緒與互動分析失敗: ${error?.message || '未知錯誤'}`);
    } finally {
      setIsAnalyzingEmotion(false);
    }
  };
  
  // AI 問答
  const handleAskQuestion = async () => {
    if (!question.trim() || speechLogs.length === 0) return;
    
    setIsAsking(true);
    setAnswer('');
    try {
      const transcript = speechLogs
        .map(l => {
          // ✅ 使用後端回傳的 speaker_nickname 或 speaker_name
          const speakerName = l.speaker_nickname || l.speaker_name || l.speaker?.slice(0, 8) || '';
          return `${speakerName ? speakerName + '：' : ''}${l.content ?? ''}`;
        })
        .filter(Boolean)
        .join('\n');
      
      const result = await aiService.askQuestion(transcript, question, 'zh-TW');
      setAnswer(result);
    } catch (error: any) {
      console.error('AI 問答失敗:', error);
      setAnswer(`⚠️ AI 問答失敗: ${error?.message || '未知錯誤'}`);
    } finally {
      setIsAsking(false);
    }
  };

  const handleClose = () => {
    setIsVisible(false);
    // 等待動畫完成後再關閉
    setTimeout(() => onClose(), 200);
  };

  const handleUpdate = async () => {
    setIsUpdating(true);
    try {
      const updates: any = {};
      if (editData.title) updates.title = editData.title;
      if (editData.session_type) updates.session_type = editData.session_type;
      if (editData.summary) updates.summary = editData.summary;

      await apiService.updateSession(session.uuid, updates);
      setIsEditing(false);
      onUpdate();
    } catch (error) {
      console.error('Failed to update session:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDelete = () => {
    onDelete();
    handleClose();
  };

  return createPortal(
    <div 
      className={`fixed inset-0 bg-black flex items-center justify-center z-50 p-4 transition-all duration-200 ease-in-out ${
        isVisible ? 'bg-opacity-50' : 'bg-opacity-0'
      }`}
      onClick={handleClose}
    >
      <div 
        className={`bg-white rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden transition-all duration-200 ease-in-out transform ${
          isVisible 
            ? 'opacity-100 scale-100 translate-y-0' 
            : 'opacity-0 scale-95 translate-y-4'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold text-gray-900">
              {session.title || `場合 ${session.uuid.slice(0, 8)}`}
            </h2>
            <button
              onClick={handleClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              ✕
            </button>
          </div>
          
          {/* Tabs */}
          <div className="flex space-x-1 mt-4">
            {[
              { id: 'info', label: '資訊', icon: Calendar },
              { id: 'logs', label: '語音記錄', icon: MessageSquare },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as any)}
                className={`
                  flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors
                  ${activeTab === id 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[calc(90vh-200px)]">
          {activeTab === 'info' && (
            <div className="space-y-6">
              {isEditing ? (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">名稱</label>
                    <input
                      type="text"
                      value={editData.title}
                      onChange={(e) => setEditData(prev => ({ ...prev, title: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">類型</label>
                    <input
                      type="text"
                      value={editData.session_type}
                      onChange={(e) => setEditData(prev => ({ ...prev, session_type: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">會議摘要</label>
                    <textarea
                      value={editData.summary}
                      onChange={(e) => setEditData(prev => ({ ...prev, summary: e.target.value }))}
                      className="input-field h-24 resize-none"
                    />
                  </div>
                  <div className="flex space-x-3">
                    <button
                      onClick={() => setIsEditing(false)}
                      className="btn-secondary transform hover:scale-105 active:scale-95 transition-transform duration-150"
                      disabled={isUpdating}
                    >
                      取消
                    </button>
                    <button
                      onClick={handleUpdate}
                      className="btn-primary transform hover:scale-105 active:scale-95 transition-transform duration-150"
                      disabled={isUpdating}
                    >
                      {isUpdating ? '更新中...' : '儲存'}
                    </button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <InfoItem label="UUID" value={session.uuid} />
                    <InfoItem label="會議ID" value={session.session_id} />
                    <InfoItem label="名稱" value={session.title} />
                    <InfoItem label="類型" value={session.session_type} />
                    {session.start_time && (
                      <InfoItem 
                        label="開始時間" 
                        value={new Date(session.start_time).toLocaleString('zh-TW')} 
                      />
                    )}
                    {session.end_time && (
                      <InfoItem 
                        label="結束時間" 
                        value={new Date(session.end_time).toLocaleString('zh-TW')} 
                      />
                    )}
                  </div>
                  
                  {session.summary && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-2">摘要</h4>
                      <p className="text-gray-700">{session.summary}</p>
                    </div>
                  )}
                  
                  {session.participants && session.participants.length > 0 && (
                    <div className="bg-gray-50 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-2 flex items-center space-x-2">
                        <Users className="w-4 h-4" />
                        <span>參與者 ({session.participants.length})</span>
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {session.participants.map((participant, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-primary-100 text-primary-700 rounded-md text-sm"
                          >
                            {participantNames[participant] || participant.slice(0, 8)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  <div className="flex space-x-3">
                    <button
                      onClick={() => setIsEditing(true)}
                      className="btn-primary flex items-center space-x-2 transform hover:scale-105 active:scale-95 transition-transform duration-150"
                    >
                      <Edit3 className="w-4 h-4" />
                      <span>編輯</span>
                    </button>
                    <button
                      onClick={handleDelete}
                      className="btn-danger flex items-center space-x-2 transform hover:scale-105 active:scale-95 transition-transform duration-150"
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>刪除</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="space-y-6">
              {/* 置頂：AI 自動摘要 (全寬) */}
              <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                    <Sparkles className="w-5 h-5 text-blue-600" />
                    <span>AI 自動摘要</span>
                  </h3>
                  {isSummarizing && (
                    <div className="flex items-center space-x-2 text-sm text-blue-600">
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                      <span>生成中...</span>
                    </div>
                  )}
                </div>
                <div className="bg-white rounded-lg p-4 min-h-[150px] prose prose-sm max-w-none">
                  {aiSummary ? (
                    <ReactMarkdown
                      components={{
                        h1: ({children}) => <h1 className="text-xl font-bold mb-3 text-gray-900">{children}</h1>,
                        h2: ({children}) => <h2 className="text-lg font-semibold mb-2 text-gray-900 border-b border-gray-300 pb-1">{children}</h2>,
                        h3: ({children}) => <h3 className="text-md font-medium mb-2 text-gray-800">{children}</h3>,
                        ul: ({children}) => <ul className="list-disc space-y-1 mb-3 ml-6">{children}</ul>,
                        ol: ({children}) => <ol className="list-decimal space-y-2 mb-3 ml-6">{children}</ol>,
                        li: ({children}) => {
                          return <li className="text-gray-700 leading-relaxed pl-1">{children}</li>;
                        },
                        p: ({children}) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
                        blockquote: ({children}) => <blockquote className="border-l-4 border-blue-400 pl-4 my-3 italic text-gray-600 bg-blue-50 py-2 rounded-r">{children}</blockquote>,
                        strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                        em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                      }}
                    >
                      {aiSummary}
                    </ReactMarkdown>
                  ) : (
                    <p className="text-gray-400 italic">{isSummarizing ? '正在生成摘要...' : '載入中...'}</p>
                  )}
                </div>
              </div>

              {/* 下方:左側文字稿 + 右側 AI 功能 */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                {/* 左側:完整逐字稿 */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                    <MessageSquare className="w-5 h-5 text-gray-600" />
                    <span>完整逐字稿</span>
                  </h3>
                  {isLoadingLogs ? (
                    <div className="flex justify-center py-8">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
                    </div>
                  ) : speechLogs.length === 0 ? (
                    <p className="text-gray-500 text-center py-8">尚無語音記錄</p>
                  ) : (
                    <div className="space-y-3">
                      {speechLogs.map(log => (
                        <div key={log.uuid} className="bg-gray-50 rounded-lg p-4">
                          {log.content && (
                            <p className="text-gray-900 mb-2">{log.content}</p>
                          )}
                          <div className="flex flex-wrap gap-4 text-xs text-gray-500">
                            {log.timestamp && (
                              <span>時間：{new Date(log.timestamp).toLocaleString('zh-TW')}</span>
                            )}
                            {typeof log.confidence === 'number' && (
                              <span>信心度：{(log.confidence * 100).toFixed(1)}%</span>
                            )}
                            {typeof log.duration === 'number' && (
                              <span>時長：{log.duration.toFixed(1)}s</span>
                            )}
                            {(log.speaker_nickname || log.speaker_name || log.speaker) && (
                              <span className="font-medium text-primary-600">
                                語者：{log.speaker_nickname || log.speaker_name || log.speaker?.slice(0, 8) || 'Unknown'}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* 右側：AI 功能 (情緒分析 + 問答) */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                    <Sparkles className="w-5 h-5 text-gray-600" />
                    <span>AI 智能分析</span>
                  </h3>
                  
                  <div className="space-y-6">
                    {/* 情緒與互動分析 */}
                    <div className="card bg-gradient-to-br from-purple-50 to-pink-50 border border-purple-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                        <Brain className="w-5 h-5 text-purple-600" />
                        <span>情緒與互動分析</span>
                      </h3>
                      <button
                        onClick={handleEmotionAnalysis}
                        disabled={isAnalyzingEmotion || speechLogs.length === 0}
                        className="btn-primary bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 flex items-center space-x-2 text-sm px-3 py-1.5"
                      >
                        {isAnalyzingEmotion ? (
                          <>
                            <div className="animate-spin rounded-full h-3 w-3 border-2 border-white border-t-transparent"></div>
                            <span>分析中...</span>
                          </>
                        ) : (
                          <>
                            <Brain className="w-3 h-3" />
                            <span>開始分析</span>
                          </>
                        )}
                      </button>
                    </div>
                    {emotionAnalysis ? (
                      <div className="bg-white rounded-lg p-4 prose prose-sm max-w-none">
                        <ReactMarkdown
                          components={{
                            h1: ({children}) => <h1 className="text-xl font-bold mb-3 text-gray-900">{children}</h1>,
                            h2: ({children}) => <h2 className="text-lg font-semibold mb-2 text-gray-900 border-b border-gray-300 pb-1">{children}</h2>,
                            h3: ({children}) => <h3 className="text-md font-medium mb-2 text-gray-800">{children}</h3>,
                            p: ({children}) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
                            ul: ({children}) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                            ol: ({children}) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                            li: ({children}) => <li className="text-gray-700">{children}</li>,
                            strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                            em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                          }}
                        >
                          {emotionAnalysis}
                        </ReactMarkdown>
                      </div>
                    ) : (
                      <div className="bg-white rounded-lg p-4 text-center">
                        <p className="text-gray-400 italic text-sm">點擊「開始分析」按鈕以進行情緒與互動分析</p>
                      </div>
                    )}
                  </div>
                  
                  {/* AI 問答 */}
                  <div className="card bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                        <MessageSquare className="w-5 h-5 text-green-600" />
                        <span>AI 問答</span>
                      </h3>
                    </div>
                    <div className="space-y-3">
                      <input
                        type="text"
                        className="input-field"
                        placeholder="輸入你的問題"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && question.trim() && !isAsking && handleAskQuestion()}
                      />
                      <div className="flex justify-end">
                        <button 
                          className="btn-primary bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-sm"
                          onClick={handleAskQuestion}
                          disabled={isAsking || !question.trim()}
                        >
                          {isAsking ? '回答中...' : '送出提問'}
                        </button>
                      </div>
                      {answer && (
                        <div className="bg-white rounded-lg p-4 prose prose-sm max-w-none">
                          <ReactMarkdown
                            components={{
                              h1: ({children}) => <h1 className="text-xl font-bold mb-3 text-gray-900">{children}</h1>,
                              h2: ({children}) => <h2 className="text-lg font-semibold mb-2 text-gray-900 border-b border-gray-300 pb-1">{children}</h2>,
                              h3: ({children}) => <h3 className="text-md font-medium mb-2 text-gray-800">{children}</h3>,
                              p: ({children}) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
                              ul: ({children}) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                              ol: ({children}) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                              li: ({children}) => <li className="text-gray-700">{children}</li>,
                              strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                              em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                            }}
                          >
                            {answer}
                          </ReactMarkdown>
                        </div>
                      )}
                      {!answer && !isAsking && (
                        <div className="bg-white rounded-lg p-4 text-center">
                          <p className="text-gray-400 italic text-sm">AI 回答將顯示於此</p>
                        </div>
                      )}
                    </div>
                  </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};

interface CreateSessionModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

const CreateSessionModal: React.FC<CreateSessionModalProps> = ({ onClose, onSuccess }) => {
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
        .map(p => p.trim())
        .filter(p => p.length > 0);

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
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl p-6 w-full max-w-md">
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
              className="btn-secondary flex-1 transform hover:scale-105 active:scale-95 transition-transform duration-150"
              disabled={isCreating}
            >
              取消
            </button>
            <button
              type="submit"
              className="btn-primary flex-1 transform hover:scale-105 active:scale-95 transition-transform duration-150"
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

interface InfoItemProps {
  label: string;
  value?: string;
}

const InfoItem: React.FC<InfoItemProps> = ({ label, value }) => {
  if (!value) return null;
  
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <dt className="text-sm font-medium text-gray-500">{label}</dt>
      <dd className="mt-1 text-sm text-gray-900 break-all">{value}</dd>
    </div>
  );
};

export default SessionsMode;