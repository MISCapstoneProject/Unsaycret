import React from 'react';
import TypewriterText from './TypewriterText';

interface IncrementalTypewriterTextProps {
  fullContent: string;
  previousContent?: string;
  speed?: number;
  pauseAtComma?: number;
}

const IncrementalTypewriterText: React.FC<IncrementalTypewriterTextProps> = ({
  fullContent,
  previousContent,
  speed = 50,
  pauseAtComma = 300
}) => {
  if (!previousContent) {
    // 第一次顯示，使用完整的打字機效果
    return (
      <TypewriterText 
        text={fullContent} 
        speed={speed}
        pauseAtComma={pauseAtComma}
      />
    );
  }

  // 計算新增的文字部分
  const newText = fullContent.substring(previousContent.length);
  
  return (
    <span>
      <span>{previousContent}</span>
      {newText && (
        <TypewriterText 
          text={newText} 
          speed={speed}
          pauseAtComma={pauseAtComma}
        />
      )}
    </span>
  );
};

export default IncrementalTypewriterText;