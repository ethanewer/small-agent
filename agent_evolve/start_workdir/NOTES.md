# Work Notes

## Generalization reminder

Changes are evaluated on a **held-out task set** the inner agent never sees.
Improvements must be general-purpose — do NOT add task-specific tips, hints, or
hard-coded solutions. Categorize failures by root cause, not by task name.

Good improvement categories:
- Planning before acting
- Verification before marking complete
- Error recovery and retry logic
- Context management (keeping instruction visible)
- Shell robustness (EOF, timeouts, background processes)
- Efficient turn usage (pivot when stuck)
- Safe file editing (syntax checks after modifications)

## Session notes template

### Failure categories observed
-

### General improvement hypothesis
-

### Changes made
-

### Validation
- Tests run:
- Benchmark rerun:
- Result:

### Expected impact on unseen tasks
-

### Next steps
-
