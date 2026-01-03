"use client"

import type React from "react"

import { useEffect, useRef, useState } from "react"
import { ChevronRight } from "lucide-react"

interface TerminalLine {
  type: "command" | "output" | "error"
  content: string
  timestamp: Date
}

interface TerminalProps {
  onExecuteCommand?: (command: string) => Promise<string>
}

export function Terminal({ onExecuteCommand }: TerminalProps) {
  const [lines, setLines] = useState<TerminalLine[]>([
    {
      type: "output",
      content: "IRE Terminal v1.0.0 - Integrated Research Environment",
      timestamp: new Date(),
    },
    {
      type: "output",
      content: "Type 'help' for available commands",
      timestamp: new Date(),
    },
  ])
  const [currentCommand, setCurrentCommand] = useState("")
  const [commandHistory, setCommandHistory] = useState<string[]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)
  const terminalEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [lines])

  const executeCommand = async (cmd: string) => {
    if (!cmd.trim()) return

    // Add command to history
    setCommandHistory((prev) => [...prev, cmd])
    setHistoryIndex(-1)

    // Add command line
    setLines((prev) => [...prev, { type: "command", content: cmd, timestamp: new Date() }])

    // Built-in commands
    let output = ""
    const trimmedCmd = cmd.trim().toLowerCase()

    if (trimmedCmd === "help") {
      output = `Available commands:
  help          - Show this help message
  clear         - Clear terminal
  echo [text]   - Print text to terminal
  ls            - List files in workspace
  python [file] - Run Python script
  node [file]   - Run Node.js script
  pwd           - Print working directory
  date          - Show current date and time`
    } else if (trimmedCmd === "clear") {
      setLines([])
      setCurrentCommand("")
      return
    } else if (trimmedCmd.startsWith("echo ")) {
      output = cmd.substring(5)
    } else if (trimmedCmd === "pwd") {
      output = "/workspace"
    } else if (trimmedCmd === "date") {
      output = new Date().toString()
    } else if (trimmedCmd === "ls") {
      output = `data/
notebooks/
scripts/
README.md
requirements.txt`
    } else if (onExecuteCommand) {
      // Use custom command handler (can integrate with Tauri backend)
      try {
        output = await onExecuteCommand(cmd)
      } catch (error) {
        output = `Error: ${error instanceof Error ? error.message : "Unknown error"}`
      }
    } else {
      output = `Command not found: ${cmd}. Type 'help' for available commands.`
    }

    // Add output
    setLines((prev) => [...prev, { type: "output", content: output, timestamp: new Date() }])
    setCurrentCommand("")
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      executeCommand(currentCommand)
    } else if (e.key === "ArrowUp") {
      e.preventDefault()
      if (commandHistory.length > 0) {
        const newIndex = historyIndex + 1
        if (newIndex < commandHistory.length) {
          setHistoryIndex(newIndex)
          setCurrentCommand(commandHistory[commandHistory.length - 1 - newIndex])
        }
      }
    } else if (e.key === "ArrowDown") {
      e.preventDefault()
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)
        setCurrentCommand(commandHistory[commandHistory.length - 1 - newIndex])
      } else if (historyIndex === 0) {
        setHistoryIndex(-1)
        setCurrentCommand("")
      }
    }
  }

  return (
    <div className="flex h-full flex-col bg-[var(--color-panel)] font-mono text-sm">
      {/* Terminal Output */}
      <div className="flex-1 overflow-auto p-4">
        {lines.map((line, index) => (
          <div key={index} className="mb-1">
            {line.type === "command" ? (
              <div className="flex items-center gap-2 text-green-400">
                <ChevronRight className="h-4 w-4" />
                <span>{line.content}</span>
              </div>
            ) : line.type === "error" ? (
              <div className="text-red-400">{line.content}</div>
            ) : (
              <div className="text-foreground">{line.content}</div>
            )}
          </div>
        ))}
        <div ref={terminalEndRef} />
      </div>

      {/* Input Line */}
      <div className="flex items-center gap-2 border-t border-border bg-[var(--color-editor)] p-2 px-4">
        <ChevronRight className="h-4 w-4 text-green-400" />
        <input
          ref={inputRef}
          type="text"
          value={currentCommand}
          onChange={(e) => setCurrentCommand(e.target.value)}
          onKeyDown={handleKeyDown}
          className="flex-1 bg-transparent text-foreground outline-none"
          placeholder="Type a command..."
          autoFocus
        />
      </div>
    </div>
  )
}
