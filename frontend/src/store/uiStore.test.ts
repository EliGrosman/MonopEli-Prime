import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import { useUIStore } from './uiStore';

// Reset store between tests
beforeEach(() => {
  act(() => {
    // Reset to initial state
    const store = useUIStore.getState();
    store.closeModal();
    store.selectProperty(null);
    store.selectPlayer(null);
    store.clearToasts();
    store.setSidebarOpen(true);
    store.setGlobalLoading(false);
  });
});

describe('uiStore', () => {
  describe('initial state', () => {
    it('has correct initial values', () => {
      const state = useUIStore.getState();
      expect(state.activeModal).toBeNull();
      expect(state.modalData).toBeNull();
      expect(state.selectedPropertyPosition).toBeNull();
      expect(state.selectedPlayerId).toBeNull();
      expect(state.toasts).toEqual([]);
      expect(state.sidebarOpen).toBe(true);
      expect(state.globalLoading).toBe(false);
      expect(state.loadingMessage).toBeNull();
    });
  });

  describe('modal actions', () => {
    it('openModal sets modal name and data', () => {
      act(() => {
        useUIStore.getState().openModal('property-details', { position: 5 });
      });

      const state = useUIStore.getState();
      expect(state.activeModal).toBe('property-details');
      expect(state.modalData).toEqual({ position: 5 });
    });

    it('openModal works without data', () => {
      act(() => {
        useUIStore.getState().openModal('confirm-purchase');
      });

      const state = useUIStore.getState();
      expect(state.activeModal).toBe('confirm-purchase');
      expect(state.modalData).toBeNull();
    });

    it('closeModal clears modal state', () => {
      act(() => {
        useUIStore.getState().openModal('property-details', { position: 5 });
        useUIStore.getState().closeModal();
      });

      const state = useUIStore.getState();
      expect(state.activeModal).toBeNull();
      expect(state.modalData).toBeNull();
    });
  });

  describe('selection actions', () => {
    it('selectProperty updates selected property position', () => {
      act(() => {
        useUIStore.getState().selectProperty(10);
      });
      expect(useUIStore.getState().selectedPropertyPosition).toBe(10);
    });

    it('selectProperty can clear selection', () => {
      act(() => {
        useUIStore.getState().selectProperty(10);
        useUIStore.getState().selectProperty(null);
      });
      expect(useUIStore.getState().selectedPropertyPosition).toBeNull();
    });

    it('selectPlayer updates selected player id', () => {
      act(() => {
        useUIStore.getState().selectPlayer(2);
      });
      expect(useUIStore.getState().selectedPlayerId).toBe(2);
    });

    it('selectPlayer can clear selection', () => {
      act(() => {
        useUIStore.getState().selectPlayer(2);
        useUIStore.getState().selectPlayer(null);
      });
      expect(useUIStore.getState().selectedPlayerId).toBeNull();
    });
  });

  describe('toast actions', () => {
    it('addToast creates a toast with generated id', () => {
      act(() => {
        useUIStore.getState().addToast('Hello World');
      });

      const toasts = useUIStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe('Hello World');
      expect(toasts[0].type).toBe('info');
      expect(toasts[0].duration).toBe(5000);
      expect(toasts[0].id).toBeTruthy();
    });

    it('addToast supports different types', () => {
      act(() => {
        useUIStore.getState().addToast('Success!', 'success');
        useUIStore.getState().addToast('Warning!', 'warning');
        useUIStore.getState().addToast('Error!', 'error');
      });

      const toasts = useUIStore.getState().toasts;
      expect(toasts[0].type).toBe('success');
      expect(toasts[1].type).toBe('warning');
      expect(toasts[2].type).toBe('error');
    });

    it('addToast supports custom duration', () => {
      act(() => {
        useUIStore.getState().addToast('Quick toast', 'info', 1000);
      });

      expect(useUIStore.getState().toasts[0].duration).toBe(1000);
    });

    it('removeToast removes specific toast', () => {
      act(() => {
        useUIStore.getState().addToast('First');
        useUIStore.getState().addToast('Second');
      });

      const toasts = useUIStore.getState().toasts;
      const firstId = toasts[0].id;

      act(() => {
        useUIStore.getState().removeToast(firstId);
      });

      const updatedToasts = useUIStore.getState().toasts;
      expect(updatedToasts).toHaveLength(1);
      expect(updatedToasts[0].message).toBe('Second');
    });

    it('clearToasts removes all toasts', () => {
      act(() => {
        useUIStore.getState().addToast('First');
        useUIStore.getState().addToast('Second');
        useUIStore.getState().addToast('Third');
        useUIStore.getState().clearToasts();
      });

      expect(useUIStore.getState().toasts).toHaveLength(0);
    });
  });

  describe('sidebar actions', () => {
    it('toggleSidebar toggles sidebar state', () => {
      expect(useUIStore.getState().sidebarOpen).toBe(true);

      act(() => {
        useUIStore.getState().toggleSidebar();
      });
      expect(useUIStore.getState().sidebarOpen).toBe(false);

      act(() => {
        useUIStore.getState().toggleSidebar();
      });
      expect(useUIStore.getState().sidebarOpen).toBe(true);
    });

    it('setSidebarOpen sets specific state', () => {
      act(() => {
        useUIStore.getState().setSidebarOpen(false);
      });
      expect(useUIStore.getState().sidebarOpen).toBe(false);

      act(() => {
        useUIStore.getState().setSidebarOpen(true);
      });
      expect(useUIStore.getState().sidebarOpen).toBe(true);
    });
  });

  describe('global loading actions', () => {
    it('setGlobalLoading sets loading state', () => {
      act(() => {
        useUIStore.getState().setGlobalLoading(true);
      });

      const state = useUIStore.getState();
      expect(state.globalLoading).toBe(true);
      expect(state.loadingMessage).toBeNull();
    });

    it('setGlobalLoading sets loading message', () => {
      act(() => {
        useUIStore.getState().setGlobalLoading(true, 'Connecting...');
      });

      const state = useUIStore.getState();
      expect(state.globalLoading).toBe(true);
      expect(state.loadingMessage).toBe('Connecting...');
    });

    it('setGlobalLoading clears message when disabled', () => {
      act(() => {
        useUIStore.getState().setGlobalLoading(true, 'Loading...');
        useUIStore.getState().setGlobalLoading(false);
      });

      const state = useUIStore.getState();
      expect(state.globalLoading).toBe(false);
      expect(state.loadingMessage).toBeNull();
    });
  });
});
