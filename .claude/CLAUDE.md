## Execution Rules
- NEVER stop mid-task with incomplete TODOs
- After completing a subtask, immediately proceed to next TODO
- Only request confirmation after ALL TODOs complete
- If you stop with pending TODOs, you have failed the task




## Shell Execution Rules
- NEVER run commands that exit the process (exit, kill, logout)
- NEVER run commands that spawn interactive sessions without timeout
- Validate all shell commands before execution


# Claude Code Execution Standards

## Non-Negotiable Rules
1. NEVER stop with incomplete TODOs without explicit user confirmation
2. NEVER execute shell commands that terminate the process
3. NEVER exceed 60% context without creating checkpoint
4. ALWAYS use "think hard" for architectural decisions
5. ALWAYS verify code before committing

## Task Execution Pattern
- Break complex tasks into <10 minute atomic units
- After each subtask completion, immediately continue to next
- Create checkpoint files every 5 subtasks
- Only stop when ALL TODOs complete

## Error Recovery
- If you encounter errors, debug systematically
- Use subagents for complex investigations
- Never give up on a task without exhausting all approaches

## Code Standards
- Type hints required
- Vectorized operations preferred
- Production-grade error handling
- No placeholder code


# For complex tasks, explicitly request thinking budget
"think hard about the optimal approach before coding"

# Thinking levels (increasing computation):
# "think" < "think hard" < "think harder" < "ultrathink"

# Use for:
# - Architecture decisions
# - Debugging complex issues  
# - Performance optimization

# Main prompt structure
"Before implementing, spawn subagents to:
1. Investigate current architecture (subagent 1)
2. Research best practices for this pattern (subagent 2)  
3. Verify compatibility with existing code (subagent 3)

After subagents report, create implementation plan for my approval."
