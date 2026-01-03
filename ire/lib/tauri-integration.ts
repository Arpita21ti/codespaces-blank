// Tauri integration utilities
// This file provides helper functions to connect your React frontend with Tauri backend

// Type definitions for Tauri invoke function
declare global {
  interface Window {
    __TAURI__?: {
      invoke: (command: string, args?: Record<string, any>) => Promise<any>
    }
  }
}

// Check if running in Tauri environment
export function isTauriEnvironment(): boolean {
  return typeof window !== "undefined" && "__TAURI__" in window
}

// Generic invoke wrapper with error handling
export async function tauriInvoke<T>(command: string, args?: Record<string, any>): Promise<T> {
  if (!isTauriEnvironment()) {
    throw new Error("Not running in Tauri environment")
  }

  try {
    const result = await window.__TAURI__!.invoke(command, args)
    return result as T
  } catch (error) {
    console.error(`[v0] Tauri invoke error for command "${command}":`, error)
    throw error
  }
}

// File system operations
export async function readFile(path: string): Promise<string> {
  return tauriInvoke<string>("read_file", { path })
}

export async function writeFile(path: string, content: string): Promise<void> {
  return tauriInvoke<void>("write_file", { path, content })
}

export async function listDirectory(path: string): Promise<string[]> {
  return tauriInvoke<string[]>("list_directory", { path })
}

// LLM integration
export async function callLLM(prompt: string, context?: string): Promise<string> {
  return tauriInvoke<string>("call_llm", { prompt, context })
}

export async function streamLLM(prompt: string, onChunk: (chunk: string) => void): Promise<void> {
  // For streaming, you'll need to set up an event listener
  // This is a placeholder implementation
  const response = await callLLM(prompt)
  onChunk(response)
}

// Terminal/Shell operations
export async function executeCommand(command: string): Promise<string> {
  return tauriInvoke<string>("execute_command", { command })
}

// Dataset operations
export async function loadDataset(path: string): Promise<any> {
  return tauriInvoke<any>("load_dataset", { path })
}

export async function queryDataset(datasetId: string, query: string): Promise<any> {
  return tauriInvoke<any>("query_dataset", { datasetId, query })
}
