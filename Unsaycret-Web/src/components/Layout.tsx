import React from 'react';
import { Mic, Users, Calendar } from 'lucide-react';
import { useIsMobile } from '../hooks/useIsMobile';

interface LayoutProps {
  children: React.ReactNode;
  currentMode: string;
  onModeChange: (mode: string) => void;
}

const Layout: React.FC<LayoutProps> = ({ children, currentMode, onModeChange }) => {
  const modes = [
    { id: 'silent', label: '主功能', icon: Mic },
    { id: 'users', label: '用戶列表', icon: Users },
    { id: 'sessions', label: '紀錄管理', icon: Calendar },
  ];

  const isMobile = useIsMobile(1024);

  // Logo 組件 - 專業麥克風版本
  const Logo = ({ mobile = false }: { mobile?: boolean }) => (
    <div className="flex items-center space-x-3">
      <div className="relative">
        <Mic className={mobile ? "w-6 h-6 text-primary-600" : "w-7 h-7 text-primary-600"} />
        <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white"></div>
      </div>
      <div className="flex flex-col items-start">
        <h1 className={mobile ? "text-lg font-bold text-gray-900" : "text-xl font-bold text-gray-900"}>Unsaycret</h1>
        {!mobile && <p className="text-[11px] text-gray-500 -mt-0.5">聲藏不漏</p>}
      </div>
    </div>
  );

  if (isMobile) {
    return (
      <div className="min-h-screen bg-white">
        {/* Mobile Header */}
        <header className="bg-white border-b border-gray-200">
          <div className="px-4 h-14 flex items-center justify-between">
            <Logo mobile />
          </div>
        </header>

        {/* Content */}
        <main className="px-3 py-3 max-w-screen-sm mx-auto">
          {children}
        </main>

        {/* Bottom Tabs */}
        <nav className="fixed bottom-0 inset-x-0 bg-white border-t border-gray-200">
          <div className="max-w-screen-sm mx-auto grid grid-cols-3">
            {modes.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => onModeChange(id)}
                className={`flex flex-col items-center py-2 text-sm ${currentMode === id ? 'text-primary-600' : 'text-gray-500'}`}
              >
                <Icon className="w-5 h-5" />
                <span className="mt-0.5">{label}</span>
              </button>
            ))}
          </div>
        </nav>
        {/* Safe area for bottom nav */}
        <div className="h-14" />
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Logo />
            
            <nav className="flex space-x-1">
              {modes.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => onModeChange(id)}
                  className={`
                    flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200
                    ${currentMode === id 
                      ? 'bg-primary-600 text-white shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                    }
                  `}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
};

export default Layout;