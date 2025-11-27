# 聲藏不漏 - 基於 AI 語音技術之多語者互動分析平台

Unsaycret 是一個整合語音轉錄、聲紋識別與語意分析的綜合平台。本專案包含後端 API 服務與前端視覺化介面，旨在提供高效的會議記錄與語者分析解決方案。

## 專案結構

本專案由兩個主要部分組成：

- **Unsaycret-API**: 後端服務，基於 FastAPI，負責語音處理、聲紋識別與資料庫管理。（[詳細說明](Unsaycret-API/README.md)）
- **Unsaycret-Web**: 前端介面，基於 Vue 3 + Vite，提供使用者操作介面。（[詳細說明](Unsaycret-Web/README.md)）

## 系統需求

在開始安裝之前，請確保您的系統滿足以下要求：

- **作業系統**: Linux, Windows, macOS
- **Python**: 3.10.11 或 3.12.11
- **Node.js**: v18+
- **Docker**: 必須安裝並啟動 (用於 Weaviate 向量資料庫)

## 安裝與啟動教學

請依照以下步驟依序完成安裝與啟動。

### 第一步：環境準備

1. **安裝 Python**
   建議使用 3.10.11 以上版本。

2. **安裝 Docker**
   請至 [Docker 官網](https://www.docker.com/products/docker-desktop/) 下載並安裝 Docker Desktop。安裝後請啟動 Docker。

### 第二步：啟動後端服務 (Unsaycret-API)

1. **進入後端目錄**
   ```bash
   cd Unsaycret-API
   ```

2. **建立虛擬環境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   ```

3. **安裝依賴套件**
   ```bash
   pip install -r requirements-base.txt
   pip install -r requirements-gpu.txt
   ```
   *注意：如果您的電腦沒有獨立顯卡，請安裝 `requirements-cpu.txt`，不要安裝 `requirements-gpu.txt`。*

4. **設定環境變數**
   複製範例設定檔並依需求修改：
   ```bash
   cp .env.example .env
   ```

5. **啟動 Docker 服務 (Weaviate)**
   ```bash
   docker-compose up -d
   ```

6. **啟動後端 API**
   ```bash
   python main.py
   ```
   成功啟動後，API 文件可於 `http://localhost:8000/docs` 存取。

### 第三步：AI 功能設定 (可選)

本系統內建 AI 智慧助手功能（摘要、問答）。**此步驟為可選**，若不需要 AI 功能可直接跳過，不影響其他功能運作。

1.  進入前端目錄 `Unsaycret-Web`。
2.  複製範例設定檔：`cp .env.example .env`
3.  編輯 `.env` 檔案，填入您的 API Key：
    ```ini
    VITE_OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    VITE_OPENAI_MODEL=gpt-4o-mini
    ```

### 第四步：啟動前端介面 (Unsaycret-Web)

1.  **開啟新的終端機視窗，進入前端目錄**
    ```bash
    cd Unsaycret-Web
    ```

2.  **安裝依賴套件**
    ```bash
    npm install
    ```

3.  **啟動開發伺服器**
    ```bash
    npm run dev
    ```
    成功啟動後，請瀏覽終端機顯示的網址 (通常為 `http://localhost:5173`)。

## 使用指南

1. **首頁儀表板**
   - 進入前端頁面後，您可以看到系統。
   
2. **語音轉錄**
   - 點擊「上傳音檔」或「開始錄音」。
   - 系統會自動進行語者分離與文字轉錄。

3. **語者管理**
   - 在轉錄結果中，您可以編輯語者名稱。
   - 系統會自動學習並建立聲紋庫。

4. **場合管理**
   - 系統會紀錄文字稿，方便後續檢索。

5. **AI 智慧助手**
   - 使用內建的 AI 功能進行會議摘要與問答。

## 開發者工具

- **Weaviate 工具**: 位於 `Unsaycret-API/weaviate_tools`，提供資料庫維護腳本。
- **API 文件**: 啟動後端後訪問 `http://localhost:8000/docs`。

## 外部連線設定 (External Access)

如果您希望讓其他設備（如手機、平板或其他電腦）存取本系統，最簡單快速的方式是使用 **Cloudflare Tunnel** 建立臨時通道。

由於本系統包含前端與後端，兩者都需要能被外部存取：

1.  **建立後端通道**：
    *   在後端終端機執行：`cloudflared tunnel --url http://localhost:8000`
    *   複製產生的 URL (例如 `https://backend-api.trycloudflare.com`)。

2.  **設定前端連接後端**：
    *   開啟 `Unsaycret-Web/src/config/api.ts`。
    *   將 `BASE_URL` 修改為上一步取得的後端通道 URL：
        ```typescript
        const BASE_URL = "https://backend-api.trycloudflare.com"; // 替換為您的後端 URL
        ```

3.  **建立前端通道**：
    *   在前端終端機執行：`cloudflared tunnel --url http://localhost:5173`
    *   複製產生的 URL (例如 `https://frontend-app.trycloudflare.com`)。

4.  **開始使用**：
    *   現在您可以使用任何設備的瀏覽器，連線到前端通道 URL (`https://frontend-app.trycloudflare.com`) 即可使用完整功能。

## 常見問題

**Q: 安裝依賴時發生錯誤？**
A: 請確認您的 Python 版本是否正確，以及是否已安裝必要的系統編譯工具。

**Q: Docker 無法啟動？**
A: 請確認 Docker Desktop 是否已開啟，且連接埠 8080 未被佔用。

---
**源代碼提交版本**
最後更新日期: 2025-11-27
