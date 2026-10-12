---
layout: post
author: Suhas Chundi
tags: [Agents]
---

## Introduction
Long horizon agent tasks are partially limited by LLM context. There are a couple simple strategies used in agent design to prevent context limits from being exceeded.

**Harness-defined**: 
* Codex summarizes the history once the context hits a length threshold
* MEM1 rewrites it at every turn

**Action-based**:
* Self-Compact and AutoCompact let the model decide when to compact
* ACM adds offloading and retrieval
* Sculptor lets the model pick which fragments to operate on

**RLMs**:
* The model executes code to constrain what information is brought into its context.


## Just make it a file
A **Context Language Model (CLM)** is allowed to edit its own context as if it was a file, with Bash or Python

<img src="/images/CLM/fig1_overview.png">

1. Models invent their own strategies: loops for pruning irrelevant search results, reusable compaction functions, etc.
2. Good zero shot performance on deep research tasks and on a large agent-swarm task.
3. Context management hints can be given as prompts
4. RL can reward trajectories that are correct and cheap.

<details>
    <summary><b>Toy example: ContextBench</b></summary>
    <img src="/images/CLM/fig2_contextbench.png">
    <ol>
        <li>Needle Retention: Can you retain needles embedded in long filler messages?</li>
        <li>Sudoku Sketchpad: Can you make edits in place?</li>
        <li>KV Store: Can you offload important but long information to disk?</li>
        <li>Log Triage: Can you operate on context programmatically?</li>
    </ol>
    <p>The LLM gets a pass only if the answer is in its context. <b>This is a toy example, and all of these problems could be solved if the constraints on ContextBench were relaxed.</b></p>
</details>


## Implementation

The CLM implementation is a thin layer on top of a minimal Bash agent (Mini-SWE-Agent). Before every command, the harness writes the agent's conversation to `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, with each turn in a header:

```
[[CTX_TURN 1 role=assistant]]
THOUGHT: Let me search for ...
bash {"command": "bcp_search 'award launched 2011'"}

[[CTX_TURN 2 role=tool]]
<10 search snippets, ~5K tokens>
...
```
In pseudocode:
```
ctx_file = ctx_file + turn_tokens + optional_ctx_edit_confirmation
new_ctx = system_prompt + tool_skill_ctx + ctx_file
```

If the ctx file was edited, the `optional_ctx_edit_confirmation` might look like `[LIVE_CTX_MAIN.txt: edit applied — context ~24100->9800 tokens, 14 turns]`. An edit that modifies the ctx file might look like:

```python
import re; p="/tmp/.live_ctx/LIVE_CTX_MAIN.txt"; s=open(p).read()
# collapse turn 7's body into a one-line note, keeping its header
s=re.sub(r"(\[\[CTX_TURN 7 [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)",
         r"\1\n[grep done: parser.py:142 drops quoted commas; fix=csv.reader]", s, flags=re.S)
open(p,"w").write(s)
```

The system prompt asks the model to locate text with code rather than retyping it.

A harness could easily extend this context as a file logic to multi-agent or subagent systems by creating context files for new agents.

The code that parses this file into the chat template is very forgiving: 
* Text outside a header becomes a user turn
* Empty turns are dropped
* Consecutive same-role turns are merged. 
* Any role other than `assistant` is mapped back to `user`

### Emergent strategies

<img src="/images/CLM/fig3_qualitative.png">

* While orchestrating subagents, it keeps an in-context scoreboard (best score, running agents, key files, dead ends), updating it in place 163 times while keeping the whole context at 6-8K tokens.
* It rewrites a span of turns into a single `notes` turn holding its findings so far.
* It writes loops that replace each old search result with a one-line stub (`[Searched: <query>]`, `[Retrieved doc <id>]`).
* It defines a `compact_turns` function once and calls it 37 times later in the same run.
* It also rediscovers what the baselines do: compressing 21K tokens into an answer-focused summary, or keeping a ledger of untried ideas.

### Compute FLOPs

Edits to context mean there is less opportunity for prefix caching. The paper computes FLOPs based on model architecture to measure this loss v.s. the reduced prompt size.

$$\text{FLOPs} = \text{FLOPs}_{\text{prefill}}(\text{tokens from the first mismatch onward}) + \text{FLOPs}_{\text{decode}}(\text{generated tokens})$$

For a single turn of Qwen3.6-27B with a 20K-token prompt and a 500-token response:

<img src="/images/CLM/fig14_prefix_reuse_flops.png">

In Qwen3.6-27B only 16 of the 64 layers use full attention. Compacting a 20K context to 5K would save only around 10% per subsequent append-only turn, while the edit itself costs a significant amount due to re-prefill. The KV cache is basically free until it isn't. Edits are FLOP efficient only when they are:

- **Infrequent and batched**: One large/compound compaction instead of many tiny improvements
- **Late in the context**: Edits near the end keep a long prefix cache
- **Solving Context Overflow**: If the alternative is summarization / autocompaction, then CLMs can win

The final point is also related to model serving hardware. If the size of the KV cache is the bottleneck, you may choose to trade more FLOPs for lower memory requirements.

<details>
    <summary><b>Suffix Cache Reuse</b></summary>
    <p>You could also just keep the cache for the suffix as well, even if its stale. Make sure you correct ROPE embeddings here as well</p>
    <img src="/images/CLM/fig4_scr.png">
    <p>A similar idea could be applied to layers that use linear attention.</p>
    <img src="/images/CLM/fig10_scr_layers.png">
    <h4>Results</h4>
    <p>On BrowseComp-Plus with Qwen3.6-27B, SCR matches standard SGLang accuracy (60.2% vs 60.2%) with 65% of the FLOPs (7.14 vs 10.98 PFLOPs per question). On turns that follow an edit, it reuses 28% of prompt tokens that would otherwise be re-prefilled.</p>
    <img src="/images/CLM/fig9_scr_results.png">
    <p><b>Most of the savings don't come from CLM edits.</b> Reasoning models' chat templates strip earlier turns' reasoning blocks, which also breaks the prefix match. Of the 7.8% of prompt tokens SCR reuses, 5.3% come from reasoning stripping and only 2.5% from context edits</p>
    <h4>When is SCR safe, and when is it a bad idea?</h4>
    <p>SCR's error is "the suffix still remembers the old <code>B</code>". So the question is <b>whether remembering <code>B</code> is harmless, helpful, or harmful.</b></p>
    <p>Edits where SCR is probably fine:</p>
    <ul>
        <li><b>Reasoning stripping.</b> The reasoning really did precede <code>C</code> when it was generated. The stale cache is arguably better</li>
        <li><b>Deleting stale, irrelevant content.</b> Pruning empty search results, dead ends, or huge tool outputs. Attention mechanisms probably skipped over this anyways.</li>
        <li><b>Replacing <code>B</code> with a summary of <code>B</code>.</b> <code>B'</code> and <code>B</code> say the same thing, so <code>C</code> is consistent with both.</li>
        <li><b>Small edits followed by a long suffix.</b> Lots of compute is saved here.</li>
    </ul>
    <p>Edits where SCR is likely harmful:</p>
    <ul>
        <li><b>Edits meant to remove bad information.</b> For example, deleting a prompt injection, a hallucinated fact, or content the model was told to forget. With SCR, the suffix KV cache or linear attention state still encodes those ideas</li>
        <li><b>Edits that contradict <code>B</code>.</b> For example, correcting a wrong hypothesis, updating state in place</li>
        <li><b>Large Edits</b> Edits that modify a lot of spans programmatically or a large chunk in the middle</li>
        <li><b>RL rollouts</b> If rollouts are served with SCR but the trainer recomputes log-probs with exact attention, the policy that generated the data isn't the one being trained on</li>
        <li><b>Other Edge Cases</b> For example, rewriting instructions, role headers, or the code that "the error above" refers to.</li>
    </ul>
    <img src="/images/CLM/Suffix_Caching.jpg">
</details>

## Improving Context Management

Obviously you can just steer with instructions / skills.

You can also carefully construct rewards around correctness and FLOPs and posttrain models using RL to become better at context management.

For all turns:
* Reward is assigned based on the success of the trajectory

For context modification turns:
* If the trajectory is unsuccessful, no additional term is added
* If the trajectory is successful, `flops_traj` is normalized against other successful trajectories, and an advantage term is added.

## Results

All of these are zero-shot CLMs (no training) compared against baselines, with a shared Mini-SWE-Agent backbone.

### Coding and deep research

Qwen3.6-27B, 32K context limit, 100-turn cap

<img src="/images/CLM/fig5_zero_shot.png">

- **BrowseComp-Plus:** Deep-research questions where answers must be pieced together by searching and reading many documents from a fixed web corpus.
- **TerminalBench 2.1:** Hard tasks in a sandboxed Linux terminal (building, debugging, configuring software)
- **TBLite:** A set of terminal-agent tasks, chosen to give quick signal while iterating on terminal agents.

\*A 32k context limit is pretty small, so these results need to be taken with a grain of salt

### Open discovery problems

These are longer horizons, from five hours up to a full day.

**Single-repository optimization** (EdgeBench-10, 12 hours, best of 3 seeds) and **multi-repository optimization** (Software World, six agents jointly optimizing interdependent Python repos for 24+ hours, scored on four downstream packages they never see):

<img src="/images/CLM/fig6_long_horizon.png">

**Math optimization** 
Claude 4.6 Sonnet is placed in the CLM harness and OpenEvolve evolutionary workflow. Again a context limit of 32k is pretty small, so these results need to be taken with a grain of salt. 

| Method | Circle packing (↑) | Heilbronn (↑) | Min-max/min-dist (↑) | Erdős overlap (↓) |
|---|---|---|---|---|
| OpenEvolve | 2.541 | 0.03127 | 0.07690 | 0.38123 |
| OpenEvolve-Agent | 2.525 | 0.03053 | 0.07724 | 0.38167 |
| CLM | 2.618 | **0.03653** | **0.07758** | **0.38094** |
| CLM (subagents) | **2.636** | 0.03617 | **0.07758** | 0.38109 |

**RL:** Qwen3.5-9B trained on OpenResearcher with the success-gated efficiency advantage, evaluated on BrowseComp-Plus:

| Method | Accuracy (%) ↑ | PFLOPs / question ↓ |
|---|---|---|
| Summary | 34.7 → 42.1 | 4.01 → 2.19 |
| CLM | 28.8 → 42.5 | 1.52 → 1.34 |

RL improves accuracy of CLMs and PFLOPs per question simultaneously.

The authors also show results where context management can be steered with instructions.