import React, { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { User, Edit3, Calendar, Search, Trash2, MessageSquare, Users, FileText, ChevronRight, Mic } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Speaker, Session, SpeechLog } from '../types';
import { apiService } from '../services/api';
import { aiService } from '../services/ai';
import { AI_CONFIG } from '../config/ai';
import { useIsMobile } from '../hooks/useIsMobile';
import { useRecording } from '../contexts/RecordingContext';

const UsersMode: React.FC = () => {
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [filteredSpeakers, setFilteredSpeakers] = useState<Speaker[]>([]);
  const [selectedSpeaker, setSelectedSpeaker] = useState<Speaker | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [genderFilter, setGenderFilter] = useState('');
  const { isRecording } = useRecording();

  const fetchSpeakers = async () => {
    setIsLoading(true);
    try {
      const fetchedSpeakers = await apiService.fetchSpeakers();
      setSpeakers(fetchedSpeakers);
      setFilteredSpeakers(fetchedSpeakers);
    } catch (error) {
      console.error('Failed to fetch speakers:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSpeakers();
  }, []);

  useEffect(() => {
    let filtered = speakers;

    if (searchTerm) {
      filtered = filtered.filter(speaker =>
        (speaker.full_name?.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (speaker.nickname?.toLowerCase().includes(searchTerm.toLowerCase())) ||
        speaker.uuid.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (genderFilter) {
      filtered = filtered.filter(speaker => speaker.gender === genderFilter);
    }

    setFilteredSpeakers(filtered);
  }, [speakers, searchTerm, genderFilter]);

  const uniqueGenders = Array.from(new Set(speakers.map(s => s.gender).filter(Boolean)));

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
      
      {/* Header Section */}
      <div className="card">
        <div className="space-y-4">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
            <User className="w-6 h-6" />
            <span>語者管理</span>
          </h2>

          <div className="flex flex-col space-y-3 sm:flex-row sm:space-y-0 sm:space-x-3">
            <div className="relative flex-1 sm:flex-initial sm:w-64">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="搜尋語者..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input-field pl-10 w-full"
              />
            </div>

            <select
              value={genderFilter}
              onChange={(e) => setGenderFilter(e.target.value)}
              className="input-field w-full sm:w-auto"
            >
              <option value="">所有性別</option>
              {uniqueGenders.map((gender) => (
                <option key={gender} value={gender}>
                  {gender}
                </option>
              ))}
            </select>

            <button
              onClick={fetchSpeakers}
              className="btn-primary w-full sm:w-auto transform hover:scale-105 active:scale-95 transition-transform duration-150"
              disabled={isLoading}
            >
              {isLoading ? '載入中...' : '重新載入'}
            </button>
          </div>
        </div>
      </div>

      {/* Speakers Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredSpeakers.map(speaker => (
            <SpeakerCard
              key={speaker.uuid}
              speaker={speaker}
              onClick={() => setSelectedSpeaker(speaker)}
            />
          ))}
        </div>
      )}

      {filteredSpeakers.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <User className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500">找不到符合條件的語者</p>
        </div>
      )}

      {/* Speaker Detail Modal */}
      {selectedSpeaker && (
        <SpeakerDetailModal
          speaker={selectedSpeaker}
          onClose={() => setSelectedSpeaker(null)}
          onUpdate={fetchSpeakers}
        />
      )}
    </div>
  );
};

interface SpeakerCardProps {
  speaker: Speaker;
  onClick: () => void;
}

const SpeakerCard: React.FC<SpeakerCardProps> = ({ speaker, onClick }) => {
  return (
    <div
      onClick={onClick}
      className="card cursor-pointer hover:shadow-lg transform hover:-translate-y-1 hover:scale-[1.02] transition-all duration-200 active:scale-[0.98]"
    >
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold text-gray-900 truncate">
          {speaker.nickname || speaker.full_name || `語者 ${speaker.uuid.slice(0, 8)}`}
        </h3>
        <Edit3 className="w-4 h-4 text-gray-400" />
      </div>
      
      <div className="space-y-2 text-sm text-gray-600">
        {(speaker.full_name || speaker.nickname) && (
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            <span>語者全名：{speaker.full_name || speaker.nickname}</span>
          </div>
        )}
        
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
          <span>
            語者性別：
            <span className={speaker.gender ? '' : 'text-gray-400'}>
              {speaker.gender || '(待設定)'}
            </span>
          </span>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
          <span>語音樣本：{speaker.voiceprint_ids.length}</span>
        </div>
      </div>
      
      {speaker.last_active_at && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            最後活躍：{new Date(speaker.last_active_at).toLocaleString('zh-TW')}
          </p>
        </div>
      )}
    </div>
  );
};

interface SpeakerDetailModalProps {
  speaker: Speaker;
  onClose: () => void;
  onUpdate: () => void;
}

const SpeakerDetailModal: React.FC<SpeakerDetailModalProps> = ({ speaker, onClose, onUpdate }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({
    full_name: speaker.full_name || '',
    nickname: speaker.nickname || '',
    gender: speaker.gender || '',
  });
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeTab, setActiveTab] = useState<'info' | 'sessions'>('info');
  const [isUpdating, setIsUpdating] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);

  // 動畫控制
  useEffect(() => {
    // 組件掛載後立即顯示動畫
    const timer = setTimeout(() => setIsVisible(true), 10);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const sessionsData = await apiService.fetchSpeakerSessions(speaker.uuid);
        setSessions(sessionsData);
      } catch (error) {
        console.error('Failed to fetch speaker data:', error);
      }
    };

    fetchData();
  }, [speaker.uuid]);

  const handleClose = () => {
    setIsVisible(false);
    // 等待動畫完成後再關閉
    setTimeout(() => onClose(), 200);
  };

  const handleUpdate = async () => {
    setIsUpdating(true);
    try {
      const updates: any = {};
      if (editData.full_name) updates.full_name = editData.full_name;
      if (editData.nickname) updates.nickname = editData.nickname;
      if (editData.gender) updates.gender = editData.gender;

      await apiService.updateSpeaker(speaker.uuid, updates);
      setIsEditing(false);
      onUpdate();
    } catch (error) {
      console.error('Failed to update speaker:', error);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await apiService.deleteSpeaker(speaker.uuid);
      onUpdate(); // 更新語者列表
      onClose(); // 關閉模態框
    } catch (error) {
      console.error('Failed to delete speaker:', error);
      alert('刪除語者失敗，請稍後重試');
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
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
              {speaker.nickname || speaker.full_name || `語者 ${speaker.uuid.slice(0, 8)}`}
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
              { id: 'info', label: '基本資料', icon: User },
              { id: 'sessions', label: '參與場合', icon: Calendar },
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
                    <label className="block text-sm font-medium text-gray-700 mb-1">全名</label>
                    <input
                      type="text"
                      value={editData.full_name}
                      onChange={(e) => setEditData(prev => ({ ...prev, full_name: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">暱稱</label>
                    <input
                      type="text"
                      value={editData.nickname}
                      onChange={(e) => setEditData(prev => ({ ...prev, nickname: e.target.value }))}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">性別</label>
                    <input
                      type="text"
                      value={editData.gender}
                      onChange={(e) => setEditData(prev => ({ ...prev, gender: e.target.value }))}
                      className="input-field"
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
                    <InfoItem label="UUID" value={speaker.uuid} />
                    <InfoItem label="語者ID" value={speaker.speaker_id?.toString()} />
                    <InfoItem label="全名" value={speaker.full_name} />
                    <InfoItem label="暱稱" value={speaker.nickname} />
                    <InfoItem label="性別" value={speaker.gender} />
                    <InfoItem label="見面次數" value={speaker.meet_count?.toString()} />
                    <InfoItem label="見面天數" value={speaker.meet_days?.toString()} />
                    <InfoItem label="語音樣本數" value={speaker.voiceprint_ids.length.toString()} />
                  </div>
                  
                  {speaker.created_at && (
                    <InfoItem 
                      label="建立時間" 
                      value={new Date(speaker.created_at).toLocaleString('zh-TW')} 
                    />
                  )}
                  
                  {speaker.last_active_at && (
                    <InfoItem 
                      label="最後活躍" 
                      value={new Date(speaker.last_active_at).toLocaleString('zh-TW')} 
                    />
                  )}
                  
                  <div className="flex space-x-3">
                    <button
                      onClick={() => setIsEditing(true)}
                      className="btn-primary flex items-center space-x-2 transform hover:scale-105 active:scale-95 transition-transform duration-150"
                    >
                      <Edit3 className="w-4 h-4" />
                      <span>編輯資料</span>
                    </button>
                    
                    <button
                      onClick={() => setShowDeleteConfirm(true)}
                      className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2 transform hover:scale-105 active:scale-95 transition-all duration-150"
                      disabled={isDeleting}
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>刪除語者</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'sessions' && (
            <div className="space-y-3">
              {sessions.length === 0 ? (
                <p className="text-gray-500 text-center py-8">尚無記錄</p>
              ) : (
                <div className="space-y-3">
                  {sessions.map(session => (
                    <div 
                      key={session.uuid} 
                      className="bg-gray-50 border border-gray-200 rounded-lg p-4 shadow-sm cursor-pointer hover:bg-white hover:shadow-md hover:border-blue-300 transition-all duration-200 group"
                      onClick={() => setSelectedSession(session)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                            {session.title || '未命名'}
                          </h4>
                          <div className="mt-2 space-y-1 text-sm text-gray-600">
                            {session.session_type && <p>類型：{session.session_type}</p>}
                            {session.start_time && (
                              <p>開始時間：{new Date(session.start_time).toLocaleString('zh-TW')}</p>
                            )}
                            {session.participants && (
                              <p>參與者：{session.participants.length} 人</p>
                            )}
                          </div>
                        </div>
                        <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 group-hover:translate-x-1 transition-all flex-shrink-0 mt-1" />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* 場合詳情模態框 */}
      {selectedSession && (
        <SessionDetailModal
          session={selectedSession}
          currentSpeaker={speaker}
          onClose={() => setSelectedSession(null)}
        />
      )}
      
      {/* 刪除確認對話框 */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-60 p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">確認刪除語者</h3>
            <p className="text-gray-600 mb-6">
              您確定要刪除語者「{speaker.nickname || speaker.full_name || `語者 ${speaker.uuid.slice(0, 8)}`}」嗎？
              <br />
              <span className="text-red-600 font-medium">此操作將永久刪除該語者及其所有聲紋資料，無法復原。</span>
            </p>
            <div className="flex space-x-3 justify-end">
              <button
                onClick={() => setShowDeleteConfirm(false)}
                className="btn-secondary"
                disabled={isDeleting}
              >
                取消
              </button>
              <button
                onClick={handleDelete}
                className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={isDeleting}
              >
                {isDeleting ? '刪除中...' : '確認刪除'}
              </button>
            </div>
          </div>
        </div>
      )}
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
      <dd className="mt-1 text-sm text-gray-900 break-all">{value || '(待設定)'}</dd>
    </div>
  );
};

// SessionDetailModal 組件（從語者頁面查看場合詳情用）
interface SessionDetailModalProps {
  session: Session;
  currentSpeaker: Speaker; // ✅ 新增: 當前查看的語者
  onClose: () => void;
}

const SessionDetailModal: React.FC<SessionDetailModalProps> = ({ session, currentSpeaker, onClose }) => {
  const isMobile = useIsMobile(1024);
  const [speechLogs, setSpeechLogs] = useState<SpeechLog[]>([]);
  const [activeTab, setActiveTab] = useState<'info' | 'logs'>('logs'); // ✅ 預設進入「語音記錄」tab
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  // ✅ 添加參與者名稱對照表
  const [participantNames, setParticipantNames] = useState<Record<string, string>>({});
  // ✅ AI 功能狀態
  const [speakerAnalysis, setSpeakerAnalysis] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // 動畫控制
  useEffect(() => {
    const timer = setTimeout(() => setIsVisible(true), 10);
    return () => clearTimeout(timer);
  }, []);

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

  useEffect(() => {
    const fetchSpeechLogs = async () => {
      setIsLoadingLogs(true);
      try {
        const logs = await apiService.fetchSessionSpeechLogs(session.uuid);
        // ✅ 只保留當前語者的發言
        const filteredLogs = logs.filter(log => log.speaker === currentSpeaker.uuid);
        setSpeechLogs(filteredLogs);
        // ✅ 後端已經包含 speaker_name 和 speaker_nickname,不需要前端查詢
      } catch (error) {
        console.error('Failed to fetch speech logs:', error);
      } finally {
        setIsLoadingLogs(false);
      }
    };

    fetchSpeechLogs();
  }, [session.uuid, currentSpeaker.uuid]);

  // ✅ AI 個人發言分析
  const handleAnalyzeSpeaker = async () => {
    if (speechLogs.length === 0) {
      setSpeakerAnalysis('該語者在此場合中沒有發言記錄');
      return;
    }

    setIsAnalyzing(true);
    setSpeakerAnalysis('');
    try {
      const transcript = speechLogs
        .map(log => log.content)
        .filter(Boolean)
        .join('\n');
      
      const speakerName = currentSpeaker.nickname || currentSpeaker.full_name || currentSpeaker.uuid.slice(0, 8);
      const result = await aiService.summarizeSpeakerContent(transcript, speakerName, { language: 'zh-TW', style: 'bullet' });
      setSpeakerAnalysis(result);
    } catch (error: any) {
      console.error('個人發言分析失敗:', error);
      setSpeakerAnalysis(`⚠️ 分析失敗: ${error?.message || '未知錯誤'}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => onClose(), 200);
  };

  return createPortal(
    <div 
      className="fixed inset-0 flex items-center justify-center p-4"
      style={{ zIndex: 9999 }}
      onClick={handleClose}
    >
      <div 
        className={`bg-white rounded-xl w-full max-w-4xl max-h-[90vh] overflow-hidden shadow-2xl transition-all duration-200 ease-in-out transform ${
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
            </div>
          )}

          {activeTab === 'logs' && (
            <div className="space-y-4">
              {/* ✅ 添加視覺提示: 只顯示該語者的發言 */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start space-x-3">
                <Users className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-blue-900">
                    僅顯示語者「{currentSpeaker.nickname || currentSpeaker.full_name}」的發言
                  </h3>
                  <p className="text-xs text-blue-700 mt-1">
                    共 {speechLogs.length} 條記錄 · 如需查看完整對話，請前往「紀錄管理」頁面
                  </p>
                </div>
              </div>
              
              {isLoadingLogs ? (
                <div className="flex justify-center py-8">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600"></div>
                </div>
              ) : speechLogs.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-gray-500 mb-2">該語者在此場合中沒有發言記錄</p>
                  <p className="text-sm text-gray-400">可能是未被識別或未參與對話</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* 左側：發言記錄列表 */}
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
                  
                  {/* 右側：AI 個人發言分析 */}
                  {!isMobile && (
                    <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 h-fit sticky top-4">
                      <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
                          <FileText className="w-5 h-5 text-blue-600" />
                          <span>個人發言分析</span>
                        </h3>
                        <button 
                          className="btn-primary bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-sm px-3 py-1.5" 
                          onClick={handleAnalyzeSpeaker} 
                          disabled={isAnalyzing || speechLogs.length === 0}
                        >
                          {isAnalyzing ? '分析中…' : '開始分析'}
                        </button>
                      </div>
                      <div className="bg-white rounded-lg p-4 min-h-[300px] max-h-[600px] overflow-y-auto prose prose-sm max-w-none">
                        {speakerAnalysis ? (
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
                              code: ({children}) => <code className="bg-gray-200 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
                            }}
                          >
                            {speakerAnalysis}
                          </ReactMarkdown>
                        ) : (
                          <div className="text-gray-400 italic flex flex-col items-center justify-center h-[280px]">
                            {AI_CONFIG.OPENAI_API_KEY ? (
                              <>
                                <FileText className="w-12 h-12 text-gray-300 mb-3" />
                                <p className="text-center">
                                  點擊「開始分析」以整理此語者的發言內容
                                </p>
                                <p className="text-xs text-gray-400 mt-2">
                                  將分析發言主題、關鍵觀點和風格特徵
                                </p>
                              </>
                            ) : (
                              <p className="text-center">⚠️ 請先填入 API Key 後使用</p>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};

export default UsersMode;