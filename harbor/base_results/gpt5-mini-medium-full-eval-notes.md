# Full Benchmark Evaluation Notes

## Results Summary

All harnesses evaluated on 83 tasks from `terminal-bench@2.0` (6 excluded).

| Harness | Passed | Failed | Max Turns | Timeout | Error | Score |
|---------|--------|--------|-----------|---------|-------|-------|
| Iter 0000 (initial) | 15 | 58 | 8 | 0 | 2 | 15/83 = **0.181** |
| Iter 0020 (asymmetric truncation) | 20 | 53 | 5 | 0 | 5 | 20/83 = **0.241** |
| Iter 0021 (keep tmux alive) | 25 | 52 | 3 | 0 | 3 | 25/83 = **0.301** |

## Passed Tasks per Harness

### Iter 0000 (15 passed)

break-filter-js-from-html, cobol-modernization, configure-git-webserver,
constraints-scheduling, custom-memory-heap-crash, distribution-search,
extract-elf, hf-model-inference, mcmc-sampling-stan, modernize-scientific-stack,
multi-source-data-merger, openssl-selfsigned-cert, prove-plus-comm,
pytorch-model-recovery, regex-log

### Iter 0020 (20 passed)

bn-fit-modify, cobol-modernization, configure-git-webserver,
constraints-scheduling, custom-memory-heap-crash, distribution-search,
fix-code-vulnerability, fix-git, git-multibranch, hf-model-inference,
merge-diff-arc-agi-task, modernize-scientific-stack, multi-source-data-merger,
nginx-request-logging, openssl-selfsigned-cert, portfolio-optimization,
prove-plus-comm, regex-log, sparql-university, sqlite-with-gcov

### Iter 0021 (25 passed)

bn-fit-modify, break-filter-js-from-html, build-cython-ext, cancel-async-tasks,
circuit-fibsqrt, cobol-modernization, constraints-scheduling,
custom-memory-heap-crash, distribution-search, extract-elf, fix-git,
headless-terminal, hf-model-inference, kv-store-grpc, merge-diff-arc-agi-task,
modernize-scientific-stack, multi-source-data-merger, nginx-request-logging,
openssl-selfsigned-cert, portfolio-optimization, prove-plus-comm, pypi-server,
regex-log, sqlite-with-gcov, vulnerable-secret

## Excluded Tasks

6 tasks are excluded from all runs due to structural environment incompatibilities
with the `WorkspaceHarborAgent` bootstrap process. These failures are deterministic
and reproduced on retry.

| Task | Reason |
|------|--------|
| `build-pmars` | Container has debian-managed `typing_extensions` that pip cannot uninstall (`no RECORD file`). Bootstrap fails. |
| `winning-avg-corewars` | Container has debian-managed `typing_extensions` that pip cannot uninstall (`no RECORD file`). Bootstrap fails. |
| `mailman` | Container has debian-managed `typing_extensions` 4.10.0 that pip cannot uninstall (`RECORD file not found`). Bootstrap fails. |
| `install-windows-3.11` | Container has debian-managed `typing_extensions` 4.10.0 that pip cannot uninstall (`RECORD file not found`). Bootstrap fails. |
| `qemu-startup` | Container has Python 3.6 which cannot install `truststore` (`No matching distribution found`). Bootstrap fails. |
| `qemu-alpine-ssh` | Container has Python 3.6 which cannot install `truststore` (`No matching distribution found`). Bootstrap fails. |

## Run Configuration

- Dataset: `terminal-bench@2.0`
- Model: `gpt-5-mini-medium`
- Timeout multiplier: 2
- Agent: `WorkspaceHarborAgent` (workspace-specific agent code)
- Concurrency: 8 (iter-0000), 16 (iter-0020, iter-0021)

## Harness Descriptions

- **Iter 0000** (initial baseline): unmodified `terminus2` agent
- **Iter 0020** (parent: iter-0014): asymmetric output truncation (1/3 head, 2/3 tail in `limit_output_length`)
- **Iter 0021** (parent: iter-0020): keep tmux session alive on successful completion (built on iter-0020's truncation change)

## Job Directories

- `harbor/jobs/full-eval-iter-0000/` -- iter-0000 main run (89 tasks, 83 evaluated)
- `harbor/jobs/full-eval-iter-0000-rerun/` -- iter-0000 rerun of 7 errored tasks (all re-failed)
- `harbor/jobs/full-eval-iter-0020/` -- iter-0020 main run (83 tasks)
- `harbor/jobs/full-eval-iter-0021/` -- iter-0021 main run (83 tasks)
