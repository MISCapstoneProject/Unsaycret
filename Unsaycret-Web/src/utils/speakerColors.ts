import { SUBTITLE_CONFIG } from '../config/subtitle';

class SpeakerColorManager {
  private speakerColors: Map<string, string> = new Map();
  private colorIndex = 0;

  /**
   * 為語者分配顏色，同一語者會返回相同的顏色
   */
  getColor(speakerName: string): string {
    if (this.speakerColors.has(speakerName)) {
      return this.speakerColors.get(speakerName)!;
    }

    // 分配新顏色
    const color = SUBTITLE_CONFIG.speakerColors[this.colorIndex % SUBTITLE_CONFIG.speakerColors.length];
    this.speakerColors.set(speakerName, color);
    this.colorIndex++;
    
    return color;
  }

  /**
   * 清除所有語者顏色記錄
   */
  reset(): void {
    this.speakerColors.clear();
    this.colorIndex = 0;
  }

  /**
   * 獲取當前已分配的語者數量
   */
  getSpeakerCount(): number {
    return this.speakerColors.size;
  }

  /**
   * 獲取所有已分配的語者及其顏色
   */
  getAllSpeakers(): { [speakerName: string]: string } {
    return Object.fromEntries(this.speakerColors);
  }
}

// 創建全域單例
export const speakerColorManager = new SpeakerColorManager();

export default speakerColorManager;