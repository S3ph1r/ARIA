# ARIA Pacing Fix - Task Description

## Objective
Implement a 60-second cooldown (silence period) AFTER each Gemini response is received and posted. This prevents model stress and avoids 503 UNAVAILABLE errors.

## Context
The current logic in `CloudManager` waits 60s since the START of the previous request. If the request takes 30s, the model only gets 30s of rest. We want 60s of rest AFTER the response.

## Instructions for the Agent

1.  **Open Workspace:** Ensure you are in the ARIA root directory on PC 139.
2.  **File to Modify:** `aria_node_controller/core/cloud_manager.py`.
3.  **Implementation:**
    *   Locate the `process_cloud_task` method.
    *   Find the line where the result is posted:
        ```python
        self.qm.post_result(task, result)
        ```
    *   Add a `time.sleep(60)` immediately AFTER this line:
        ```python
        self.qm.post_result(task, result)
        time.sleep(60)  # Pacing cooldown (ensure 1 min of silence after response)
        ```
    *   Ensure `import time` is present at the top of the file (it should be).
4.  **Verification:**
    *   Start ARIA Node.
    *   Submit 2-3 cloud tasks.
    *   Verify in logs that there is a clear 60s gap between "Cloud task completed" of one job and the "requesting global pacing slot" of the next.

## Post-Fix Action
Once confirmed, notify the user so we can resume the DIAS pipeline by removing the `dias:status:paused` flag in Redis.
