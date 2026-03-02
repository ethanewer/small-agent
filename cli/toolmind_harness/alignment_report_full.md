# Harness Alignment Report

## Static Checks
- Prompt similarity (normalized, date-agnostic): `0.575`
- Parser round-trip success on reference assistant turns: `1.000`

## Reference Aggregate
- count: `2.000`
- avg_turns: `214.000`
- avg_assistant_turns: `106.500`
- avg_think_ratio: `1.000`
- avg_tool_ratio: `0.978`
- avg_one_tool_ratio: `0.978`
- avg_protocol_repairs: `0.000`
- avg_assistant_chars: `1472.281`

## Candidate Aggregate
- count: `3.000`
- avg_turns: `15.333`
- avg_assistant_turns: `7.333`
- avg_think_ratio: `1.000`
- avg_tool_ratio: `0.855`
- avg_one_tool_ratio: `0.855`
- avg_protocol_repairs: `1.333`
- avg_assistant_chars: `633.787`

## Candidate Per-File
- `/Users/ethanewer/scratch/generated_trajectories_v3/traj_v3_clean_q2.json`
  - turns=19, assistant=9, think_ratio=1.000, tool_ratio=0.889, one_tool_ratio=0.889, repairs=4
  - servers={'search_and_scrape_webpage': 3, 'jina_scrape_llm_summary': 1}
  - tools={'google_search': 7, 'scrape_and_extract_info': 1}
- `/Users/ethanewer/scratch/generated_trajectories_v3/traj_v3_clean_q3.json`
  - turns=11, assistant=5, think_ratio=1.000, tool_ratio=0.800, one_tool_ratio=0.800, repairs=0
  - servers={'search_and_scrape_webpage': 3, 'jina_scrape_llm_summary': 1}
  - tools={'google_search': 3, 'scrape_and_extract_info': 1}
- `/Users/ethanewer/scratch/generated_trajectories_v3/traj_v3_retrycheck_q3.json`
  - turns=16, assistant=8, think_ratio=1.000, tool_ratio=0.875, one_tool_ratio=0.875, repairs=0
  - servers={'search_and_scrape_webpage': 3, 'jina_scrape_llm_summary': 3}
  - tools={'google_search': 3, 'scrape_and_extract_info': 4}
