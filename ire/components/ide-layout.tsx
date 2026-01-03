"use client"

import { useState } from "react"
import { ActivityBar } from "./activity-bar"
import { Sidebar } from "./sidebar"
import { EditorArea } from "./editor-area"
import { Panel } from "./panel"
import { StatusBar } from "./status-bar"

interface EditorTab {
  path: string
  name: string
  language: string
  content: string
}

export function IDELayout() {
  const [activeView, setActiveView] = useState<"explorer" | "search" | "ai" | "extensions">("explorer")
  const [sidebarVisible, setSidebarVisible] = useState(true)
  const [panelVisible, setPanelVisible] = useState(true)
  const [openFiles, setOpenFiles] = useState<EditorTab[]>([])

  const handleFileSelect = async (path: string) => {
    // Check if file is already open
    if (openFiles.some((f) => f.path === path)) {
      return
    }

    // Determine language from file extension
    const extension = path.split(".").pop() || ""
    const languageMap: Record<string, string> = {
      py: "python",
      js: "javascript",
      ts: "typescript",
      tsx: "typescript",
      jsx: "javascript",
      json: "json",
      md: "markdown",
      csv: "plaintext",
      txt: "plaintext",
      ipynb: "json",
      html: "html",
      css: "css",
      sql: "sql",
      r: "r",
      sh: "shell",
    }

    // Sample content - In real app, this would use Tauri's fs API
    // Example: const content = await invoke('read_file', { path });
    const fileName = path.split("/").pop() || path
    const sampleContent = getSampleContent(fileName, extension)

    const newFile: EditorTab = {
      path,
      name: fileName,
      language: languageMap[extension] || "plaintext",
      content: sampleContent,
    }

    setOpenFiles([...openFiles, newFile])
  }

  const handleFileClose = (path: string) => {
    setOpenFiles(openFiles.filter((f) => f.path !== path))
  }

  const handleSendMessage = async (message: string): Promise<string> => {
    // This is where you integrate your LLM API
    // Example with Tauri:
    // const response = await invoke('call_llm', { prompt: message });
    // return response;

    // For now, return structured placeholder
    await new Promise((resolve) => setTimeout(resolve, 500)) // Simulate API delay

    return `Processing your request: "${message}"\n\nTo connect your LLM:\n1. Add your API endpoint in lib/llm-client.ts\n2. Use Tauri's invoke() to call your backend\n3. Handle streaming responses for better UX\n\nExample code is in LLM_INTEGRATION.md`
  }

  const handleExecuteCommand = async (command: string): Promise<string> => {
    // This is where you integrate with Tauri's shell
    // Example:
    // const output = await invoke('execute_command', { cmd: command });
    // return output;

    return `Executed: ${command}\n\nTo enable real command execution:\n1. Use Tauri's shell API\n2. Configure allowed commands in tauri.conf.json\n3. Handle async processes properly`
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Activity Bar */}
      <ActivityBar
        activeView={activeView}
        onViewChange={(view) => {
          setActiveView(view)
          setSidebarVisible(true)
        }}
      />

      {/* Sidebar */}
      {sidebarVisible && (
        <Sidebar activeView={activeView} onClose={() => setSidebarVisible(false)} onFileSelect={handleFileSelect} />
      )}

      {/* Main Content Area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Editor */}
        <EditorArea openFiles={openFiles} onFileClose={handleFileClose} />

        {/* Bottom Panel (AI Chat, Terminal, etc.) */}
        {panelVisible && (
          <Panel
            onClose={() => setPanelVisible(false)}
            onSendMessage={handleSendMessage}
            onExecuteCommand={handleExecuteCommand}
          />
        )}
      </div>

      {/* Status Bar */}
      <StatusBar />
    </div>
  )
}

function getSampleContent(fileName: string, extension: string): string {
  const templates: Record<string, string> = {
    py: `# ${fileName}
# Python script for data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Main function for data processing"""
    # Load your dataset
    df = pd.read_csv('data/dataset.csv')
    
    # Perform analysis
    print(df.head())
    print(df.describe())
    
    # TODO: Add your analysis code here
    
if __name__ == "__main__":
    main()
`,
    js: `// ${fileName}
// JavaScript module

export function processData(data) {
  // TODO: Implement data processing logic
  return data.map(item => ({
    ...item,
    processed: true
  }));
}

export default processData;
`,
    md: `# ${fileName}

## Overview

This is a research document.

## Contents

- Introduction
- Methodology
- Results
- Conclusion

## Getting Started

Add your content here...
`,
    json: `{
  "name": "${fileName.replace(".json", "")}",
  "version": "1.0.0",
  "description": "Data configuration",
  "data": []
}`,
    csv: `id,name,value,category
1,Sample A,100,Type 1
2,Sample B,200,Type 2
3,Sample C,150,Type 1
`,
  }

  return templates[extension] || `# ${fileName}\n\n# This file is ready to edit\n# Start adding your content here\n`
}
