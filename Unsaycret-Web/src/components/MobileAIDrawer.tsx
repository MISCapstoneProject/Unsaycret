import React, { useState } from 'react';
import { PanelRightOpen, X } from 'lucide-react';
import AIPanel from './AIPanel';

interface MobileAIDrawerProps {
  outputs: string[];
  title?: string;
}

const MobileAIDrawer: React.FC<MobileAIDrawerProps> = ({ outputs, title = 'AI 助手' }) => {
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* Trigger Button */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-20 right-4 z-40 inline-flex items-center space-x-2 px-4 py-2 rounded-full bg-primary-600 text-white shadow-lg"
      >
        <PanelRightOpen className="w-5 h-5" />
        <span>AI</span>
      </button>

      {/* Drawer */}
      {open && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black bg-opacity-40"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 right-0 w-full max-w-md bg-white shadow-xl flex flex-col">
            <div className="p-4 border-b border-gray-200 flex items-center justify-between">
              <h3 className="text-base font-semibold text-gray-900">{title}</h3>
              <button onClick={() => setOpen(false)} className="text-gray-500 hover:text-gray-700">
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 overflow-y-auto">
              <AIPanel outputs={outputs} />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default MobileAIDrawer;


