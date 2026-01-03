"use client"

import { useRef } from "react"
import Editor, { type OnMount } from "@monaco-editor/react"
import type { editor } from "monaco-editor"

interface MonacoEditorWrapperProps {
  value: string
  language: string
  onChange?: (value: string | undefined) => void
  readOnly?: boolean
}

export function MonacoEditorWrapper({ value, language, onChange, readOnly = false }: MonacoEditorWrapperProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)

  const handleEditorDidMount: OnMount = (editor) => {
    editorRef.current = editor
    editor.focus()
  }

  return (
    <Editor
      height="100%"
      language={language}
      value={value}
      onChange={onChange}
      theme="vs-dark"
      onMount={handleEditorDidMount}
      options={{
        readOnly,
        minimap: { enabled: true },
        fontSize: 14,
        lineNumbers: "on",
        rulers: [80, 120],
        wordWrap: "off",
        automaticLayout: true,
        scrollBeyondLastLine: false,
        padding: { top: 10, bottom: 10 },
        fontFamily: "Geist Mono, monospace",
        fontLigatures: true,
        cursorBlinking: "smooth",
        smoothScrolling: true,
        contextmenu: true,
        quickSuggestions: true,
        suggestOnTriggerCharacters: true,
      }}
    />
  )
}
