import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface Toast {
  id: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  duration?: number;
}

interface UIStore {
  // Modal state
  activeModal: string | null;
  modalData: unknown;

  // Selection state
  selectedPropertyPosition: number | null;
  selectedPlayerId: number | null;

  // Toast notifications
  toasts: Toast[];

  // Sidebar
  sidebarOpen: boolean;

  // Loading states
  globalLoading: boolean;
  loadingMessage: string | null;

  // Actions
  openModal: (name: string, data?: unknown) => void;
  closeModal: () => void;
  selectProperty: (position: number | null) => void;
  selectPlayer: (playerId: number | null) => void;
  addToast: (message: string, type?: Toast['type'], duration?: number) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  setGlobalLoading: (loading: boolean, message?: string) => void;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 9);
}

export const useUIStore = create<UIStore>()(
  devtools(
    (set) => ({
      activeModal: null,
      modalData: null,
      selectedPropertyPosition: null,
      selectedPlayerId: null,
      toasts: [],
      sidebarOpen: true,
      globalLoading: false,
      loadingMessage: null,

      openModal: (name, data = null) =>
        set({
          activeModal: name,
          modalData: data,
        }),

      closeModal: () =>
        set({
          activeModal: null,
          modalData: null,
        }),

      selectProperty: (position) =>
        set({
          selectedPropertyPosition: position,
        }),

      selectPlayer: (playerId) =>
        set({
          selectedPlayerId: playerId,
        }),

      addToast: (message, type = 'info', duration = 5000) =>
        set((state) => ({
          toasts: [...state.toasts, { id: generateId(), message, type, duration }],
        })),

      removeToast: (id) =>
        set((state) => ({
          toasts: state.toasts.filter((t) => t.id !== id),
        })),

      clearToasts: () => set({ toasts: [] }),

      toggleSidebar: () =>
        set((state) => ({
          sidebarOpen: !state.sidebarOpen,
        })),

      setSidebarOpen: (open) => set({ sidebarOpen: open }),

      setGlobalLoading: (loading, message) =>
        set({
          globalLoading: loading,
          loadingMessage: message ?? null,
        }),
    }),
    { name: 'ui-store' }
  )
);
