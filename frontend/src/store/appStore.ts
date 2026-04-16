import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { DeviceAnalysisResult, Step, Control } from "../api";

export type Session = {
  id: string;
  imageDataUrl: string | null;
  analysis: DeviceAnalysisResult | null;
  messages: Message[];
  instructions: Step[];
  createdAt: number;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
};

interface AppState {
  sessions: Session[];
  currentSessionId: string | null;
  isAnalyzing: boolean;
  isExplaining: boolean;
  isGeneratingInstructions: boolean;
  error: string | null;
  status: string;

  createSession: (imageDataUrl: string) => string;
  setCurrentSession: (id: string) => void;
  updateSessionAnalysis: (sessionId: string, analysis: DeviceAnalysisResult) => void;
  addMessage: (sessionId: string, message: Omit<Message, "id" | "timestamp">) => void;
  setInstructions: (sessionId: string, steps: Step[]) => void;
  setAnalyzing: (value: boolean) => void;
  setExplaining: (value: boolean) => void;
  setGeneratingInstructions: (value: boolean) => void;
  setError: (error: string | null) => void;
  setStatus: (status: string) => void;
  deleteSession: (id: string) => void;
  getCurrentSession: () => Session | null;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 15);
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,
      isAnalyzing: false,
      isExplaining: false,
      isGeneratingInstructions: false,
      error: null,
      status: "Загрузите изображение устройства для анализа.",

      createSession: (imageDataUrl: string) => {
        const id = generateId();
        const session: Session = {
          id,
          imageDataUrl,
          analysis: null,
          messages: [],
          instructions: [],
          createdAt: Date.now(),
        };
        set((state) => ({
          sessions: [session, ...state.sessions].slice(0, 10),
          currentSessionId: id,
          status: "Изображение загружено. Теперь можно запустить анализ.",
        }));
        return id;
      },

      setCurrentSession: (id: string) => {
        set({ currentSessionId: id });
      },

      updateSessionAnalysis: (sessionId: string, analysis: DeviceAnalysisResult) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, analysis } : s
          ),
          status: "Анализ завершен. Теперь можно запросить объяснение или инструкции.",
        }));
      },

      addMessage: (sessionId: string, message: Omit<Message, "id" | "timestamp">) => {
        const fullMessage: Message = {
          ...message,
          id: generateId(),
          timestamp: Date.now(),
        };
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, messages: [...s.messages, fullMessage] } : s
          ),
        }));
      },

      setInstructions: (sessionId: string, steps: Step[]) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId ? { ...s, instructions: steps } : s
          ),
          status: "Инструкции готовы.",
        }));
      },

      setAnalyzing: (value: boolean) => {
        set({
          isAnalyzing: value,
          error: value ? null : get().error,
        });
        if (value) {
          set({ status: "Анализируем изображение устройства..." });
        }
      },

      setExplaining: (value: boolean) => {
        set({ isExplaining: value });
        if (value) {
          set({ status: "Генерируем объяснение...", error: null });
        }
      },

      setGeneratingInstructions: (value: boolean) => {
        set({ isGeneratingInstructions: value });
        if (value) {
          set({ status: "Генерируем пошаговые инструкции...", error: null });
        }
      },

      setError: (error: string | null) => {
        set({ error, isAnalyzing: false, isExplaining: false, isGeneratingInstructions: false });
      },

      setStatus: (status: string) => {
        set({ status });
      },

      deleteSession: (id: string) => {
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== id);
          return {
            sessions: newSessions,
            currentSessionId:
              state.currentSessionId === id
                ? newSessions[0]?.id ?? null
                : state.currentSessionId,
          };
        });
      },

      getCurrentSession: () => {
        const state = get();
        return state.sessions.find((s) => s.id === state.currentSessionId) ?? null;
      },
    }),
    {
      name: "visual-sommelier-storage",
      partialize: (state) => ({
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
      }),
    }
  )
);
