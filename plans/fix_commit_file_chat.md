# Fix Plan: `commit_file` Chat Behavior

The objective is to ensure that when Titan is asked to commit a file via chat, it performs the same chunking and master copy storage as the manual button upload.

## Analysis
- **Manual Upload**: The frontend ([`AetherChatApp.jsx`](../aether-frontend/src/AetherChatApp.jsx)) reads the file, chunks it, and calls `executeTitanCommand` for each chunk, then once more for the master copy.
- **Chat Trigger**: The backend ([`titan.py`](titan.py)) detects `[COMMIT_FILE]` in the AI's response. It then looks at the *user's original message* (`memory_text`) to find the data to commit. If the user just said "commit this file" without providing the content in that specific message, the backend only saves that sentence.

## Solution
We need to ensure that the content being "burned" to the core is the actual file content, not just the chat message triggering it.

### 1. Frontend: `AetherChatApp.jsx`
- When a file is selected, the frontend currently handles the "Anchor (Core)" mode by reading and chunking.
- If the user is in "Analyze (Chat)" mode but triggers a `[COMMIT_FILE]` via the chat interface, the frontend needs to be aware of the staged file and handle the chunking if Titan decides to commit it.
- **Better Approach**: Modify `handleSend` to include file content in the context if a file is staged, and ensure the backend knows how to extract it.

### 2. Backend: `titan.py`
- Improve the `[COMMIT_FILE]` trigger logic in `unified_titan_endpoint`.
- The backend needs to distinguish between "committing the conversation" and "committing the artifact".
- When `[COMMIT_FILE]` is triggered, it should check if there's structured file data provided in the payload or if it needs to extract a specific block from the `memory_text`.

## Implementation Steps

### Step 1: Frontend (`AetherChatApp.jsx`)
- Update `handleSend` logic for `manualCommitType === 'file'`.
- If a file is staged (`file` is not null), read its content and send it along with the trigger.

### Step 2: Backend (`titan.py`)
- Update the `[COMMIT_FILE]` block in `unified_titan_endpoint`.
- Implement a more robust "shard and master" logic that handles the extracted file content.
- Ensure `ai_score` is applied to all shards.

## Mermaid Flow

```mermaid
graph TD
    User[Architect] -->|Chat: Commit this file| Frontend[AetherChatApp]
    Frontend -->|Check if File Staged| Staged{File Staged?}
    Staged -->|Yes| ReadFile[Read File Content]
    ReadFile --> SendWithContent[Send Trigger + File Content]
    Staged -->|No| SendTrigger[Send Trigger Only]
    SendWithContent --> Backend[Titan Backend]
    SendTrigger --> Backend
    Backend -->|Detect [COMMIT_FILE]| Sharding[Chunk Content]
    Sharding -->|Loop| CommitShards[Commit Shards to DB]
    CommitShards --> CommitMaster[Commit Master Copy to DB]
    CommitMaster --> Response[Signal Anchored]
```

## Proposed Todo List
1.  [ ] Modify `AetherChatApp.jsx`: `handleSend` to include file content when `[COMMIT_FILE]` is detected and a file is staged.
2.  [ ] Modify `titan.py`: Refactor `[COMMIT_FILE]` trigger to handle multi-shard + master copy commits more reliably.
3.  [ ] Test with a medium-sized file (> 2000 chars) via chat.
