import React, { useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Sparkles, MessageCircle } from 'lucide-react';
import { aiService } from '../services/ai';
import { AI_CONFIG } from '../config/ai';

interface AIPanelProps {
  outputs: string[]; // 轉錄行
}

const AIPanel: React.FC<AIPanelProps> = ({ outputs }) => {
  const transcript = useMemo(() => outputs.join('\n'), [outputs]);
  const [summary, setSummary] = useState<string>('');
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const keyMissing = !AI_CONFIG.OPENAI_API_KEY;

  const handleSummarize = async () => {
    if (keyMissing) {
      setSummary('⚠️ 尚未設定 API Key，請至 src/config/ai.ts 填入 OPENAI_API_KEY');
      return;
    }
    if (!transcript.trim()) {
      setSummary('目前沒有可摘要的轉錄內容');
      return;
    }
    setIsSummarizing(true);
    setSummary('');
    try {
      const res = await aiService.summarize(transcript, { language: 'zh-TW', style: 'bullet' });
      setSummary(res);
    } catch (e: any) {
      setSummary(e?.message || '摘要失敗');
    } finally {
      setIsSummarizing(false);
    }
  };

  const handleAsk = async () => {
    if (keyMissing) {
      setAnswer('⚠️ 尚未設定 API Key，請至 src/config/ai.ts 填入 OPENAI_API_KEY');
      return;
    }
    if (!question.trim()) return;
    if (!transcript.trim()) {
      setAnswer('目前沒有可參考的逐字稿');
      return;
    }
    setIsAsking(true);
    setAnswer('');
    try {
      const res = await aiService.askQuestion(transcript, question, 'zh-TW');
      setAnswer(res);
    } catch (e: any) {
      setAnswer(e?.message || '提問失敗');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* AI 摘要 */}
      <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <Sparkles className="w-5 h-5 text-blue-600" />
            <span>AI 摘要</span>
          </h3>
          <button 
            className="btn-primary bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400" 
            onClick={handleSummarize} 
            disabled={isSummarizing}
          >
            {isSummarizing ? '產生中…' : '產生摘要'}
          </button>
        </div>
        <div className="bg-white rounded-lg p-4 min-h-[120px] prose prose-sm max-w-none">
          {summary ? (
            <ReactMarkdown
              components={{
                h1: ({children}) => <h1 className="text-xl font-bold mb-3 text-gray-900">{children}</h1>,
                h2: ({children}) => <h2 className="text-lg font-semibold mb-2 text-gray-900 border-b border-gray-300 pb-1">{children}</h2>,
                h3: ({children}) => <h3 className="text-md font-medium mb-2 text-gray-800">{children}</h3>,
                ul: ({children}) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                li: ({children}) => <li className="text-gray-700">{children}</li>,
                p: ({children}) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
                blockquote: ({children}) => <blockquote className="border-l-4 border-blue-400 pl-4 my-3 italic text-gray-600 bg-blue-50 py-2 rounded-r">{children}</blockquote>,
                strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                code: ({children}) => <code className="bg-gray-200 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
              }}
            >
              {summary}
            </ReactMarkdown>
          ) : (
            <div className="text-gray-400 italic">
              {keyMissing ? '⚠️ 請先填入 API Key 後使用' : '按下「產生摘要」開始'}
            </div>
          )}
        </div>
      </div>

      {/* AI 問答 */}
      <div className="card bg-gradient-to-br from-green-50 to-emerald-50 border border-green-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <MessageCircle className="w-5 h-5 text-green-600" />
            <span>AI 問答</span>
          </h3>
        </div>
        <div className="space-y-3">
          <input
            type="text"
            className="input-field"
            placeholder={keyMissing ? '請先於 src/config/ai.ts 設定 API Key' : '輸入你的問題'}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isAsking && question.trim() && handleAsk()}
            disabled={keyMissing}
          />
          <div className="flex justify-end">
            <button 
              className="btn-primary bg-green-600 hover:bg-green-700 disabled:bg-gray-400" 
              onClick={handleAsk} 
              disabled={isAsking || keyMissing || !question.trim()}
            >
              {isAsking ? '回答中…' : '送出提問'}
            </button>
          </div>
          <div className="bg-white rounded-lg p-4 min-h-[120px] prose prose-sm max-w-none">
            {answer ? (
              <ReactMarkdown
                components={{
                  h1: ({children}) => <h1 className="text-xl font-bold mb-3 text-gray-900">{children}</h1>,
                  h2: ({children}) => <h2 className="text-lg font-semibold mb-2 text-gray-900 border-b border-gray-300 pb-1">{children}</h2>,
                  h3: ({children}) => <h3 className="text-md font-medium mb-2 text-gray-800">{children}</h3>,
                  ul: ({children}) => <ul className="list-disc list-inside space-y-1 mb-2">{children}</ul>,
                  ol: ({children}) => <ol className="list-decimal list-inside space-y-1 mb-2">{children}</ol>,
                  li: ({children}) => <li className="text-gray-700">{children}</li>,
                  p: ({children}) => <p className="mb-2 text-gray-700 leading-relaxed">{children}</p>,
                  blockquote: ({children}) => <blockquote className="border-l-4 border-green-400 pl-4 my-3 italic text-gray-600 bg-green-50 py-2 rounded-r">{children}</blockquote>,
                  strong: ({children}) => <strong className="font-semibold text-gray-900">{children}</strong>,
                  em: ({children}) => <em className="italic text-gray-600">{children}</em>,
                  code: ({children}) => <code className="bg-gray-200 px-1 py-0.5 rounded text-sm font-mono">{children}</code>,
                }}
              >
                {answer}
              </ReactMarkdown>
            ) : (
              <div className="text-gray-400 italic">
                {keyMissing ? '⚠️ 請先填入 API Key 後使用' : 'AI 回答將顯示於此'}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIPanel;


