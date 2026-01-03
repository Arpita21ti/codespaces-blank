"use client"

import { Files, Search, Bot, Blocks } from "lucide-react"
import { cn } from "@/lib/utils"

type View = "explorer" | "search" | "ai" | "extensions"

interface ActivityBarProps {
  activeView: View
  onViewChange: (view: View) => void
}

export function ActivityBar({ activeView, onViewChange }: ActivityBarProps) {
  const items: { id: View; icon: typeof Files; label: string }[] = [
    { id: "explorer", icon: Files, label: "Explorer" },
    { id: "search", icon: Search, label: "Search" },
    { id: "ai", icon: Bot, label: "AI Assistant" },
    { id: "extensions", icon: Blocks, label: "Extensions" },
  ]

  return (
    <div className="flex w-12 flex-col items-center bg-[var(--color-activitybar)] py-2">
      {items.map((item) => {
        const Icon = item.icon
        const isActive = activeView === item.id

        return (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={cn(
              "relative mb-2 flex h-12 w-12 items-center justify-center transition-colors hover:bg-[var(--color-hover)]",
              isActive && "text-[var(--color-activitybar-foreground)]",
              !isActive && "text-muted-foreground",
            )}
            title={item.label}
          >
            <Icon className="h-6 w-6" />
            {isActive && <div className="absolute left-0 h-full w-0.5 bg-primary" />}
          </button>
        )
      })}
    </div>
  )
}
