import React, { useState, useEffect } from 'react';

interface TypewriterTextProps {
  text: string;
  speed?: number; // 打字速度（毫秒）
  pauseAtComma?: number; // 遇到逗號時的停頓時間（毫秒）
  onComplete?: () => void;
}

const TypewriterText: React.FC<TypewriterTextProps> = ({
  text,
  speed = 50,
  pauseAtComma = 300,
  onComplete
}) => {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (currentIndex < text.length && !isComplete) {
      const currentChar = text[currentIndex];
      // 支援更多中英文標點符號
      const isPunctuation = /[，,。.！!？?；;：:]/.test(currentChar);
      
      const delay = isPunctuation ? pauseAtComma : speed;
      
      const timer = setTimeout(() => {
        setDisplayedText(prev => prev + currentChar);
        setCurrentIndex(prev => prev + 1);
        
        if (currentIndex === text.length - 1) {
          setIsComplete(true);
          onComplete?.();
        }
      }, delay);

      return () => clearTimeout(timer);
    }
  }, [currentIndex, text, speed, pauseAtComma, onComplete, isComplete]);

  // 重置當文字改變時
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsComplete(false);
  }, [text]);

  return (
    <span className="typewriter-text">
      {displayedText}
      {!isComplete && (
        <span className="animate-pulse">|</span>
      )}
    </span>
  );
};

export default TypewriterText;