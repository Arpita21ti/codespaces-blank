"use client"

import { X } from "lucide-react"
import { FileExplorer } from "./file-explorer"

type View = "explorer" | "search" | "ai" | "extensions"

interface SidebarProps {
  activeView: View
  onClose: () => void
  onFileSelect?: (path: string) => void
}

export function Sidebar({ activeView, onClose, onFileSelect }: SidebarProps) {
  const getTitleByView = (view: View) => {
    const titles = {
      explorer: "EXPLORER",
      search: "SEARCH",
      ai: "AI ASSISTANT",
      extensions: "EXTENSIONS",
    }
    return titles[view]
  }

  return (
    <div className="flex w-64 flex-col border-r border-border bg-[var(--color-sidebar)]">
      {/* Sidebar Header */}
      <div className="flex h-9 items-center justify-between border-b border-border px-3">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {getTitleByView(activeView)}
        </span>
        <button onClick={onClose} className="text-muted-foreground transition-colors hover:text-foreground">
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Sidebar Content */}
      <div className="flex-1 overflow-auto">
        {activeView === "explorer" && <FileExplorer onFileSelect={onFileSelect} />}
        {activeView === "search" && (
          <div className="p-2 text-sm text-muted-foreground">Search functionality coming soon</div>
        )}
        {activeView === "ai" && <div className="p-2 text-sm text-muted-foreground">AI assistant panel coming soon</div>}
        {activeView === "extensions" && <div className="p-2 text-sm text-muted-foreground">Extensions coming soon</div>}
      </div>
    </div>
  )
}
