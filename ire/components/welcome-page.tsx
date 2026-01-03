"use client"

import {
  FileText,
  FolderOpen,
  Command,
  BookOpen,
  Github,
  Youtube,
  Sparkles,
  Terminal,
  Database,
  Code2,
} from "lucide-react"

export function WelcomePage() {
  return (
    <div className="flex h-full flex-col overflow-auto bg-[var(--color-editor)] p-12">
      <div className="mx-auto w-full max-w-5xl">
        {/* Header */}
        <div className="mb-12">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-blue-500/10">
              <Sparkles className="h-7 w-7 text-blue-400" />
            </div>
            <h1 className="text-4xl font-bold text-foreground">IRE</h1>
          </div>
          <p className="text-xl text-muted-foreground">Integrated Research Environment</p>
          <p className="mt-2 text-sm text-muted-foreground">AI-powered workspace for data science and research</p>
        </div>

        {/* Main Grid */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Start Section */}
          <div>
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Start</h2>
            <div className="flex flex-col gap-2">
              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <FileText className="h-5 w-5 text-blue-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">New File</div>
                  <div className="text-xs text-muted-foreground">Create a new file or notebook</div>
                </div>
                <kbd className="rounded bg-background px-2 py-1 text-xs text-muted-foreground">Ctrl+N</kbd>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <FolderOpen className="h-5 w-5 text-blue-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Open Folder</div>
                  <div className="text-xs text-muted-foreground">Open a project workspace</div>
                </div>
                <kbd className="rounded bg-background px-2 py-1 text-xs text-muted-foreground">Ctrl+O</kbd>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Database className="h-5 w-5 text-blue-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Load Dataset</div>
                  <div className="text-xs text-muted-foreground">Import CSV, JSON, or Parquet files</div>
                </div>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Command className="h-5 w-5 text-blue-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Command Palette</div>
                  <div className="text-xs text-muted-foreground">Access all commands</div>
                </div>
                <kbd className="rounded bg-background px-2 py-1 text-xs text-muted-foreground">Ctrl+Shift+P</kbd>
              </button>
            </div>
          </div>

          {/* Recent Section */}
          <div>
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Recent</h2>
            <div className="flex flex-col gap-2">
              <div className="rounded-lg border border-border bg-[var(--color-card)] p-3">
                <div className="text-xs text-muted-foreground">No recent projects</div>
              </div>
            </div>
          </div>

          {/* Learn Section */}
          <div>
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Learn</h2>
            <div className="flex flex-col gap-2">
              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <BookOpen className="h-5 w-5 text-green-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Documentation</div>
                  <div className="text-xs text-muted-foreground">Learn how to use IRE</div>
                </div>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Terminal className="h-5 w-5 text-green-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Interactive Tutorial</div>
                  <div className="text-xs text-muted-foreground">Get started with a guided tour</div>
                </div>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Code2 className="h-5 w-5 text-green-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Example Projects</div>
                  <div className="text-xs text-muted-foreground">Explore sample workflows</div>
                </div>
              </button>
            </div>
          </div>

          {/* Help Section */}
          <div>
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-muted-foreground">Help</h2>
            <div className="flex flex-col gap-2">
              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Github className="h-5 w-5 text-purple-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">View on GitHub</div>
                  <div className="text-xs text-muted-foreground">Contribute to the project</div>
                </div>
              </button>

              <button className="flex items-center gap-3 rounded-lg border border-border bg-[var(--color-card)] p-3 text-left transition-colors hover:bg-[var(--color-hover)]">
                <Youtube className="h-5 w-5 text-purple-400" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-foreground">Video Tutorials</div>
                  <div className="text-xs text-muted-foreground">Watch how-to guides</div>
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Features Highlight */}
        <div className="mt-12 grid gap-4 md:grid-cols-3">
          <div className="rounded-lg border border-border bg-[var(--color-card)] p-4">
            <div className="mb-2 text-blue-400">
              <Sparkles className="h-6 w-6" />
            </div>
            <h3 className="mb-1 text-sm font-semibold text-foreground">Talk to your data</h3>
            <p className="text-xs text-muted-foreground">
              Query, transform, and explore datasets in plain English using AI
            </p>
          </div>

          <div className="rounded-lg border border-border bg-[var(--color-card)] p-4">
            <div className="mb-2 text-blue-400">
              <Database className="h-6 w-6" />
            </div>
            <h3 className="mb-1 text-sm font-semibold text-foreground">Visualize instantly</h3>
            <p className="text-xs text-muted-foreground">
              Generate beautiful interactive charts and insights with a single command
            </p>
          </div>

          <div className="rounded-lg border border-border bg-[var(--color-card)] p-4">
            <div className="mb-2 text-blue-400">
              <Code2 className="h-6 w-6" />
            </div>
            <h3 className="mb-1 text-sm font-semibold text-foreground">Code meets clarity</h3>
            <p className="text-xs text-muted-foreground">
              A powerful research platform that feels as intuitive as a conversation
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
