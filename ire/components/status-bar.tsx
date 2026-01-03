"use client"

export function StatusBar() {
  return (
    <div className="fixed bottom-0 left-0 right-0 flex h-6 items-center justify-between bg-[var(--color-statusbar)] px-2 text-xs text-white">
      <div className="flex items-center gap-4">
        <span className="font-semibold">IRE</span>
        <span>Ready</span>
      </div>
      <div className="flex items-center gap-4">
        <span>Python 3.11</span>
        <span>UTF-8</span>
        <span>LLM Connected</span>
      </div>
    </div>
  )
}
