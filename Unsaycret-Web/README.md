# Unsaycret-Web (前端介面)

這是 **聲藏不漏** 專案的網頁前端部分，採用 **Vue 3** (或是 React，根據檔案副檔名 .tsx 判斷應為 React，請確認) + **Vite** + **TypeScript** 開發，並使用 **Tailwind CSS** 進行樣式設計。

> **注意**: 專案根目錄的 [README.md](../README.md) 包含完整的系統安裝與啟動教學。本文件僅針對前端部分的架構與開發細節進行說明。

## 📂 專案架構

本資料夾的檔案結構如下：

```
Unsaycret-Web/
├── public/              # 靜態資源檔案 (圖片、圖示等)
├── src/                 # 原始碼目錄
│   ├── components/      # UI 元件 (按功能分類)
│   ├── config/          # 設定檔 (如 API 端點設定)
│   ├── contexts/        # React Contexts (全域狀態管理)
│   ├── hooks/           # Custom Hooks (共用邏輯)
│   ├── services/        # API 服務 (與後端溝通的介面)
│   ├── types/           # TypeScript 型別定義
│   ├── utils/           # 工具函式 (Helper functions)
│   ├── App.tsx          # 主應用程式元件
│   ├── main.tsx         # 程式進入點
│   └── index.css        # 全域樣式 (Tailwind imports)
├── index.html           # HTML 入口檔案
├── package.json         # 專案依賴與腳本設定
├── tailwind.config.js   # Tailwind CSS 設定
├── tsconfig.json        # TypeScript 設定
└── vite.config.ts       # Vite 建置設定
```

## ✨ 核心功能

### 🎙️ 即時語音轉錄
- **即時串流模式**：透過 WebSocket 實現低延遲的語音轉文字
- **多語者識別**：自動識別不同說話者，並以顏色區分顯示
- **動態字幕效果**：打字機效果逐字顯示，支援標點符號停頓
- **語者顏色管理**：每位語者自動分配專屬顏色，提升可讀性

### 📋 場合管理（Sessions）
- **建立會議場合**：設定會議名稱、類型、參與者
- **場合列表**：瀏覽所有會議記錄，支援搜尋與篩選
- **詳細檢視**：查看會議摘要、參與者資訊、完整逐字稿
- **語音記錄**：檢視每段發言的時間戳、信心度、發言時長

### 👥 語者管理（Users）
- **語者檔案**：管理語者基本資料（全名、暱稱、性別）
- **智能搜尋**：支援名稱、UUID 搜尋，性別篩選
- **語者統計**：查看見面次數、活躍時間、語音樣本數
- **關聯記錄**：檢視語者參與的所有會議與發言記錄
- **語者刪除**：安全的語者資料刪除功能（含確認對話框）

### 🤖 AI 智慧助手
- **自動摘要**：將會議逐字稿整理成結構化要點
- **智能問答**：針對會議內容進行問答互動
- **多語言支援**：支援中文、英文等多種語言輸出
- **行動版優化**：手機端提供抽屜式 AI 介面

### 📱 響應式設計
- **桌面版**：上方導覽列切換功能模組
- **行動版**：底部 Tab 導覽，觸控友好
- **PWA 支援**：可安裝至手機桌面，原生 App 體驗

## 🛠️ 技術架構

- **前端框架**：React 18 + TypeScript
- **樣式設計**：TailwindCSS + Lucide Icons
- **狀態管理**：React Hooks
- **即時通訊**：WebSocket（自定義 `useASRWebStreamBare` Hook）
- **AI 整合**：OpenAI API（GPT 模型）
- **建置工具**：Vite
- **其他功能**：
  - React Markdown 渲染 AI 回答
  - PWA 支援（manifest + service worker）
  - 增量式打字機效果組件

## 🚀 開發指南

### 1. 安裝依賴
```bash
npm install
```

### 2. 啟動開發伺服器
```bash
npm run dev
```
預設會運行在 `http://localhost:5173`。

### 3. 建置生產版本
```bash
npm run build
```
建置後的檔案會輸出至 `dist/` 目錄。

## 🛠️ 技術棧

- **核心框架**: React 18 + TypeScript
- **建置工具**: Vite
- **樣式框架**: Tailwind CSS
- **HTTP 請求**: Axios (位於 services 目錄)
- **狀態管理**: React Context API

## 📝 重要檔案說明

- **`src/config`**: 包含 API URL 等環境設定，開發時請確保指向正確的後端位址。
- **`src/services`**: 所有與後端 API 的互動邏輯都封裝在此，例如語者識別、轉錄請求等。
- **`src/components`**: 包含錄音介面、語者列表、轉錄結果顯示等核心 UI 元件。

## 🤖 AI 功能設定

若要啟用系統中的 AI 智慧助手（會議摘要、問答），請依照以下步驟設定：

1.  進入 `Unsaycret-Web` 目錄。
2.  複製範例設定檔：
    ```bash
    cp .env.example .env
    # Windows 使用者請用: copy .env.example .env
    ```
3.  編輯 `.env` 檔案，填入您的 OpenAI API Key：
    ```ini
    VITE_OPENAI_API_KEY=sk-proj-xxxxxxxx...
    VITE_OPENAI_MODEL=gpt-4o-mini
    ```