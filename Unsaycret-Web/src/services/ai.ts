import { AI_CONFIG } from '../config/ai';

type SummarizeOptions = {
  language?: string; // e.g., 'zh-TW'
  style?: 'bullet' | 'paragraph';
};

class AIService {
  private get headers() {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (AI_CONFIG.OPENAI_API_KEY) {
      headers['Authorization'] = `Bearer ${AI_CONFIG.OPENAI_API_KEY}`;
    }
    return headers;
  }

  private ensureKey() {
    if (!AI_CONFIG.OPENAI_API_KEY) {
      throw new Error('缺少 OpenAI API Key，請在 src/config/ai.ts 填入 OPENAI_API_KEY');
    }
  }

  async summarize(transcript: string, options: SummarizeOptions = {}) {
    this.ensureKey();
    const system = `你是一個專業的會議記錄分析師,擅長從討論內容中提煉核心重點。請用 ${
      options.language || 'zh-TW'
    } 輸出,${options.style === 'bullet' ? '使用條列式結構' : '使用段落式'}。

    請按照以下結構整理會議內容:

    1. **會議參與者與背景**:
      - 列出參與者(使用實際姓名或代號)
      - 說明會議時間與主要目的
      - 簡述會議主要議題或背景

    2. **討論主題與觀點**:
      - 歸納出 2-10 個核心討論主題
      - 每個主題下列出關鍵論點與代表性發言
      - 特別標記有爭議或需要進一步討論的觀點
      - 保留重要的數據、日期、具體方案等細節

    3. **總結**:
      - 本次會議的主要共識或結論
      - 提出的行動項目或後續待辦事項
      - 未解決的問題或需要追蹤的議題

    注意事項:
    - 保持客觀中立,如實反映各方觀點
    - 重視具體細節(數據、時間、方法等)而非空泛描述
    - 若會議中提到專有名詞或技術術語,請保留原文
    - 對於重要決策或承諾,請特別標註`;


    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: AI_CONFIG.OPENAI_MODEL,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `以下為會議逐字稿，請摘要：\n\n${transcript}` },
        ],
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`AI 摘要失敗: ${errText}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }

  /**
   * 為單一語者的發言進行內容整理
   * 適用於從語者管理頁面查看個人發言記錄的場景
   */
  async summarizeSpeakerContent(transcript: string, speakerName: string, options: SummarizeOptions = {}) {
    this.ensureKey();
    const system = `你是一個專業的內容分析師。當前需要整理單一語者「${speakerName}」在某個場合中的發言內容。請用 ${
      options.language || 'zh-TW'
    } 輸出，${options.style === 'bullet' ? '使用條列方式' : '使用段落'}。重點如下：
    
    1. **發言主題歸納**: 這個人主要談論了哪些話題？
    2. **關鍵觀點提取**: 他/她的核心論點或重要意見是什麼？
    3. **發言風格分析**: 簡述發言特點（如：簡潔、詳細、質疑、支持等）
    4. **重要資訊標記**: 特別注意提到的數據、日期、人名、決策等
    
    注意：這是單一語者的發言記錄，不是完整對話，所以不需要分析互動或爭論，專注於內容整理即可。`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: AI_CONFIG.OPENAI_MODEL,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `以下是語者「${speakerName}」的發言記錄，請整理分析：\n\n${transcript}` },
        ],
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`AI 內容整理失敗: ${errText}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }

  async askQuestion(transcript: string, question: string, language: string = 'zh-TW') {
    this.ensureKey();
    const system = `你是會議助手。根據用戶提供的會議逐字稿回答問題；若答案不在逐字稿中，請誠實說明並提供可能的追問建議。以 ${language} 回答。`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: AI_CONFIG.OPENAI_MODEL,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `逐字稿：\n${transcript}\n\n問題：${question}` },
        ],
        temperature: 0.2,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`AI 問答失敗: ${errText}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }

  async analyzeEmotion(transcript: string, language: string = 'zh-TW') {
    this.ensureKey();
    const system = `你是一位專業的對話情緒與互動分析師，擅長從對話內容中識別情緒走向、參與者情緒狀態和潛在衝突。請用 ${language} 輸出，分析以下維度：

      1. **整體情緒走向**：分析對話從開始到結束的氣氛變化（如：緊張→放鬆、消極→積極）
      2. **個別參與者情緒**：識別每位參與者的情緒特徵（如：積極、消極、焦慮、自信等）
      3. **衝突與爭議檢測**：標示對話中的爭論點、意見分歧或緊張時刻
      4. **參與度分析**：評估各參與者的投入程度（主導、被動、迴避等）
      5. **決策效率**：判斷對話是否有效推進議題（高效、冗長、偏題等）

      請以清晰的結構化格式輸出，使用標題、項目符號和重點標記。`;

    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify({
        model: AI_CONFIG.OPENAI_MODEL,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: `請分析以下對話的情緒與互動模式：\n\n${transcript}` },
        ],
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`AI 情緒與互動分析失敗: ${errText}`);
    }
    const data = await response.json();
    return data.choices?.[0]?.message?.content?.trim() || '';
  }
}

export const aiService = new AIService();


