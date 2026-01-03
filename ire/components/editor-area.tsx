"use client"

import { useState, useEffect } from "react"
import { X } from "lucide-react"
import { MonacoEditorWrapper } from "./monaco-editor-wrapper"
import { WelcomePage } from "./welcome-page"

interface EditorTab {
  path: string
  name: string
  language: string
  content: string
}

interface EditorAreaProps {
  openFiles?: EditorTab[]
  onFileClose?: (path: string) => void
}

export function EditorArea({ openFiles = [], onFileClose }: EditorAreaProps) {
  const [activeTab, setActiveTab] = useState<string | null>(null)
  const [fileContents, setFileContents] = useState<Record<string, string>>({})

  // Update active tab when files change
  useEffect(() => {
    if (openFiles.length > 0 && !activeTab) {
      setActiveTab(openFiles[0].path)
    } else if (openFiles.length === 0) {
      setActiveTab(null)
    }
  }, [openFiles, activeTab])

  // Initialize file contents when files are opened
  useEffect(() => {
    const newContents: Record<string, string> = {}
    openFiles.forEach((file) => {
      if (!fileContents[file.path]) {
        newContents[file.path] = file.content
      }
    })
    if (Object.keys(newContents).length > 0) {
      setFileContents((prev) => ({ ...prev, ...newContents }))
    }
  }, [openFiles])

  const activeFile = openFiles.find((f) => f.path === activeTab)

  const handleContentChange = (path: string, value: string | undefined) => {
    if (value !== undefined) {
      setFileContents((prev) => ({ ...prev, [path]: value }))
    }
  }

  const handleTabClose = (path: string) => {
    onFileClose?.(path)

    // Switch to another tab if the closed tab was active
    if (activeTab === path) {
      const currentIndex = openFiles.findIndex((f) => f.path === path)
      const remainingFiles = openFiles.filter((f) => f.path !== path)

      if (remainingFiles.length > 0) {
        // Switch to the next tab, or the previous one if this was the last
        const nextIndex = Math.min(currentIndex, remainingFiles.length - 1)
        setActiveTab(remainingFiles[nextIndex].path)
      } else {
        setActiveTab(null)
      }
    }
  }

  return (
    <div className="flex flex-1 flex-col overflow-hidden bg-[var(--color-editor)]">
      {/* Tab Bar */}
      {openFiles.length > 0 ? (
        <>
          <div className="flex h-9 items-center gap-px border-b border-border bg-[var(--color-card)]">
            {openFiles.map((file) => (
              <div
                key={file.path}
                className={`flex items-center gap-2 border-r border-border px-3 py-1.5 cursor-pointer hover:bg-[var(--color-hover)] transition-colors ${
                  activeTab === file.path ? "bg-[var(--color-editor)]" : "bg-[var(--color-card)]"
                }`}
                onClick={() => setActiveTab(file.path)}
              >
                <span className="text-xs text-foreground">{file.name}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleTabClose(file.path)
                  }}
                  className="text-muted-foreground hover:text-foreground hover:bg-[var(--color-hover)] rounded p-0.5 transition-colors"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>

          {/* Editor Content */}
          {activeFile && (
            <div className="flex-1">
              <MonacoEditorWrapper
                value={fileContents[activeFile.path] || activeFile.content}
                language={activeFile.language}
                onChange={(value) => handleContentChange(activeFile.path, value)}
              />
            </div>
          )}
        </>
      ) : (
        <WelcomePage />
      )}
    </div>
  )
}
