import { useState, useCallback } from 'react';

export type MobileTab = 'actions' | 'properties' | 'players' | 'log';

/**
 * Hook to manage mobile panel state.
 * Handles which panel is open and provides callbacks.
 */
export function useMobileControls() {
  const [activeTab, setActiveTab] = useState<MobileTab>('actions');
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const openPanel = useCallback((tab: MobileTab) => {
    setActiveTab(tab);
    setIsPanelOpen(true);
  }, []);

  const closePanel = useCallback(() => {
    setIsPanelOpen(false);
  }, []);

  const handleTabChange = useCallback(
    (tab: MobileTab) => {
      if (activeTab === tab && isPanelOpen) {
        closePanel();
      } else {
        openPanel(tab);
      }
    },
    [activeTab, isPanelOpen, closePanel, openPanel]
  );

  return {
    activeTab,
    isPanelOpen,
    openPanel,
    closePanel,
    handleTabChange,
  };
}

export default useMobileControls;
