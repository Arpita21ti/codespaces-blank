"use client"

import { useState } from "react"
import { X } from "lucide-react"
import { AIChat } from "./ai-chat"
import { Terminal } from "./terminal"

interface PanelProps {
  onClose: () => void
  onSendMessage?: (message: string) => Promise<string>
  onExecuteCommand?: (command: string) => Promise<string>
}

export function Panel({ onClose, onSendMessage, onExecuteCommand }: PanelProps) {
  const [activeTab, setActiveTab] = useState<"chat" | "terminal" | "output">("chat")

  return (
    <div className="flex h-64 flex-col border-t border-border bg-[var(--color-panel)]">
      {/* Panel Header */}
      <div className="flex h-9 items-center justify-between border-b border-border bg-[var(--color-card)] px-3">
        <div className="flex gap-4">
          <button
            onClick={() => setActiveTab("chat")}
            className={`text-xs font-semibold transition-colors hover:text-foreground ${
              activeTab === "chat" ? "text-blue-400" : "text-muted-foreground"
            }`}
          >
            AI CHAT
          </button>
          <button
            onClick={() => setActiveTab("terminal")}
            className={`text-xs font-semibold transition-colors hover:text-foreground ${
              activeTab === "terminal" ? "text-blue-400" : "text-muted-foreground"
            }`}
          >
            TERMINAL
          </button>
          <button
            onClick={() => setActiveTab("output")}
            className={`text-xs font-semibold transition-colors hover:text-foreground ${
              activeTab === "output" ? "text-blue-400" : "text-muted-foreground"
            }`}
          >
            OUTPUT
          </button>
        </div>
        <button onClick={onClose} className="text-muted-foreground transition-colors hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Panel Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === "chat" && <AIChat onSendMessage={onSendMessage} />}
        {activeTab === "terminal" && <Terminal onExecuteCommand={onExecuteCommand} />}
        {activeTab === "output" && (
          <div className="flex h-full items-center justify-center p-4">
            <div className="text-center text-sm text-muted-foreground">
              <p>Output panel - Shows program output and logs</p>
              <p className="mt-2 text-xs">Run scripts to see output here</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
