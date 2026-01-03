"use client"

import { useState } from "react"
import { ChevronRight, ChevronDown, File, Folder, FolderOpen } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileNode {
  name: string
  type: "file" | "folder"
  children?: FileNode[]
  path: string
}

interface FileExplorerProps {
  onFileSelect?: (path: string) => void
}

export function FileExplorer({ onFileSelect }: FileExplorerProps) {
  // Sample file structure
  const [fileTree] = useState<FileNode[]>([
    {
      name: "project-workspace",
      type: "folder",
      path: "/project-workspace",
      children: [
        {
          name: "data",
          type: "folder",
          path: "/project-workspace/data",
          children: [
            { name: "dataset.csv", type: "file", path: "/project-workspace/data/dataset.csv" },
            { name: "analysis.json", type: "file", path: "/project-workspace/data/analysis.json" },
          ],
        },
        {
          name: "notebooks",
          type: "folder",
          path: "/project-workspace/notebooks",
          children: [
            { name: "exploration.ipynb", type: "file", path: "/project-workspace/notebooks/exploration.ipynb" },
            { name: "model.py", type: "file", path: "/project-workspace/notebooks/model.py" },
          ],
        },
        {
          name: "scripts",
          type: "folder",
          path: "/project-workspace/scripts",
          children: [
            { name: "preprocess.py", type: "file", path: "/project-workspace/scripts/preprocess.py" },
            { name: "visualize.py", type: "file", path: "/project-workspace/scripts/visualize.py" },
          ],
        },
        { name: "README.md", type: "file", path: "/project-workspace/README.md" },
        { name: "requirements.txt", type: "file", path: "/project-workspace/requirements.txt" },
      ],
    },
  ])

  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set(["/project-workspace"]))

  const toggleFolder = (path: string) => {
    const newExpanded = new Set(expandedFolders)
    if (newExpanded.has(path)) {
      newExpanded.delete(path)
    } else {
      newExpanded.add(path)
    }
    setExpandedFolders(newExpanded)
  }

  const renderNode = (node: FileNode, depth = 0) => {
    const isExpanded = expandedFolders.has(node.path)
    const isFolder = node.type === "folder"

    return (
      <div key={node.path}>
        <div
          className={cn(
            "flex items-center gap-1 py-0.5 cursor-pointer hover:bg-[var(--color-hover)] text-sm",
            "text-foreground",
          )}
          style={{ paddingLeft: `${depth * 12 + 4}px` }}
          onClick={() => {
            if (isFolder) {
              toggleFolder(node.path)
            } else {
              onFileSelect?.(node.path)
            }
          }}
        >
          {isFolder ? (
            <>
              {isExpanded ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
              {isExpanded ? (
                <FolderOpen className="h-4 w-4 text-blue-400" />
              ) : (
                <Folder className="h-4 w-4 text-blue-400" />
              )}
            </>
          ) : (
            <>
              <span className="w-3.5" />
              <File className="h-4 w-4 text-muted-foreground" />
            </>
          )}
          <span className="truncate">{node.name}</span>
        </div>
        {isFolder && isExpanded && node.children && (
          <div>{node.children.map((child) => renderNode(child, depth + 1))}</div>
        )}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-1">
      <div className="px-2 py-1">
        <div className="mb-2 flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Workspace</span>
        </div>
        {fileTree.map((node) => renderNode(node))}
      </div>
    </div>
  )
}
