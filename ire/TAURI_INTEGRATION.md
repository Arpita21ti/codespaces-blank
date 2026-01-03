# Tauri Integration Guide for IRE

This guide explains how to integrate your IRE React frontend with Tauri for desktop deployment and how to connect your custom LLM.

## Prerequisites

- Node.js 18+ installed
- Rust installed (for Tauri): https://www.rust-lang.org/tools/install
- Your LLM API endpoint ready

## Step 1: Initialize Tauri in Your Project

```bash
# Install Tauri CLI
npm install --save-dev @tauri-apps/cli

# Initialize Tauri
npm install @tauri-apps/api
npx tauri init
```

During initialization:
- App name: `IRE - Integrated Research Environment`
- Window title: `Integrated Research Environment`
- Web assets location: `out` (for Next.js static export)
- Dev server URL: `http://localhost:3000`
- Frontend dev command: `npm run dev`
- Frontend build command: `npm run build`

## Step 2: Configure Next.js for Static Export

Update your `next.config.js`:

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Disable API routes for Tauri
  experimental: {
    appDir: true,
  },
}

module.exports = nextConfig
```

Update `package.json` scripts:

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "tauri": "tauri",
    "tauri:dev": "tauri dev",
    "tauri:build": "tauri build"
  }
}
```

## Step 3: Configure Tauri

Edit `src-tauri/tauri.conf.json`:

```json
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:3000",
    "distDir": "../out"
  },
  "package": {
    "productName": "IRE",
    "version": "0.1.0"
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "fs": {
        "all": true,
        "scope": ["$APPDATA/*", "$HOME/*"]
      },
      "http": {
        "all": true,
        "scope": ["http://localhost:*", "https://*"]
      },
      "shell": {
        "all": false,
        "open": true
      },
      "dialog": {
        "all": true
      }
    },
    "bundle": {
      "active": true,
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ],
      "identifier": "com.ire.app",
      "targets": "all"
    },
    "security": {
      "csp": null
    },
    "windows": [
      {
        "fullscreen": false,
        "resizable": true,
        "title": "IRE - Integrated Research Environment",
        "width": 1400,
        "height": 900,
        "minWidth": 800,
        "minHeight": 600
      }
    ]
  }
}
```

## Step 4: Integrate Your LLM API

### Option A: Local LLM (Recommended for Privacy)

If your LLM runs locally, use Tauri's HTTP client to communicate:

Create `lib/llm-client.ts`:

```typescript
import { invoke } from '@tauri-apps/api/tauri'
import { fetch } from '@tauri-apps/api/http'

export interface LLMMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface LLMResponse {
  content: string
  model: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

export async function sendToLLM(
  messages: LLMMessage[],
  options?: {
    temperature?: number
    maxTokens?: number
    model?: string
  }
): Promise<LLMResponse> {
  try {
    // Replace with your actual LLM endpoint
    const response = await fetch('http://localhost:8000/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: {
        messages,
        temperature: options?.temperature || 0.7,
        max_tokens: options?.maxTokens || 2000,
        model: options?.model || 'your-model-name',
      },
    })

    if (response.ok && response.data) {
      const data = response.data as any
      return {
        content: data.choices[0].message.content,
        model: data.model,
        usage: data.usage,
      }
    } else {
      throw new Error(`LLM API error: ${response.status}`)
    }
  } catch (error) {
    console.error('Error calling LLM:', error)
    throw error
  }
}

// Streaming version for real-time responses
export async function streamFromLLM(
  messages: LLMMessage[],
  onChunk: (chunk: string) => void,
  options?: {
    temperature?: number
    maxTokens?: number
    model?: string
  }
): Promise<void> {
  // Implement streaming based on your LLM's API
  // This will depend on whether your LLM supports SSE, WebSockets, etc.
}
```

### Option B: Remote LLM API

For cloud-based LLMs:

```typescript
export async function sendToLLM(
  messages: LLMMessage[],
  apiKey: string
): Promise<LLMResponse> {
  const response = await fetch('https://api.your-llm-provider.com/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: {
      messages,
      temperature: 0.7,
      max_tokens: 2000,
    },
  })

  // Process response...
}
```

## Step 5: Update AI Chat Component

Update `components/ide-layout.tsx` to use the LLM client:

```typescript
import { sendToLLM, type LLMMessage } from '@/lib/llm-client'

// In handleSendMessage:
const handleSendMessage = async (message: string): Promise<string> => {
  try {
    const messages: LLMMessage[] = [
      {
        role: 'system',
        content: 'You are a helpful AI assistant specialized in data science and research.'
      },
      {
        role: 'user',
        content: message
      }
    ]

    const response = await sendToLLM(messages)
    return response.content
  } catch (error) {
    console.error('Error communicating with LLM:', error)
    return 'Sorry, I encountered an error communicating with the AI model. Please check your connection and try again.'
  }
}
```

## Step 6: File System Integration

For reading/writing files, create `lib/file-system.ts`:

```typescript
import { readTextFile, writeTextFile, readDir } from '@tauri-apps/api/fs'
import { open } from '@tauri-apps/api/dialog'

export async function openFile(): Promise<{ path: string; content: string } | null> {
  try {
    const selected = await open({
      multiple: false,
      filters: [{
        name: 'All Files',
        extensions: ['*']
      }]
    })

    if (selected && typeof selected === 'string') {
      const content = await readTextFile(selected)
      return { path: selected, content }
    }
    return null
  } catch (error) {
    console.error('Error opening file:', error)
    return null
  }
}

export async function saveFile(path: string, content: string): Promise<boolean> {
  try {
    await writeTextFile(path, content)
    return true
  } catch (error) {
    console.error('Error saving file:', error)
    return false
  }
}

export async function listDirectory(path: string) {
  try {
    const entries = await readDir(path)
    return entries
  } catch (error) {
    console.error('Error reading directory:', error)
    return []
  }
}
```

## Step 7: Build and Run

### Development:
```bash
npm run tauri:dev
```

### Production Build:
```bash
npm run tauri:build
```

The built application will be in `src-tauri/target/release/bundle/`

## Step 8: Android Support (Future)

Tauri 2.0+ supports mobile platforms. To add Android support:

```bash
npm install @tauri-apps/cli@next
npx tauri android init
npx tauri android dev
```

## Additional Features to Implement

### 1. Data Visualization with Python
Integrate Python scripts for data analysis:

```rust
// In src-tauri/src/main.rs
#[tauri::command]
fn run_python_script(script: String) -> Result<String, String> {
    use std::process::Command;
    
    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .output()
        .map_err(|e| e.to_string())?;
    
    String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())
}
