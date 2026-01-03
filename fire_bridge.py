import sys
import json
import fire  # Importing your original script as a module
import io

# Redirect stdout/stderr to capture tool output during execution
# We only want to print the final JSON response to real stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

def main():
    # Load resources once (Warmup)
    # This keeps the LLM in RAM so it doesn't reload every message
    fire.get_llm() 
    fire.get_rag()
    
    # Send "READY" signal to Tauri
    print(json.dumps({"status": "ready"}), flush=True)

    while True:
        try:
            # Read line from Tauri
            line = sys.stdin.readline()
            if not line:
                break
            
            data = json.loads(line)
            command = data.get("command")
            payload = data.get("payload", {})

            response = {}

            if command == "chat":
                # Call your agent logic
                user_text = payload.get("text")
                attachments = [] # Parse attachments if sent
                
                # Capture the agent response
                result = fire.fire_agent(user_text, attachments)
                response = {"status": "ok", "data": result}

            elif command == "load_csv":
                path = payload.get("path")
                # Call tool directly
                res = fire.tool_load_csv("user_csv", path)
                response = {"status": "ok", "data": res}
            
            # ... Add handlers for other specific needs ...

            else:
                response = {"status": "error", "message": "Unknown command"}

            # Send JSON back to Tauri
            print(json.dumps(response), flush=True)

        except Exception as e:
            err = {"status": "error", "message": str(e)}
            print(json.dumps(err), flush=True)

if __name__ == "__main__":
    main()
