# SETUP — LoRAdapter Mentor in Claude Code

## A. Where the files go (inside your cloned LoRAdapter repo)
```
LoRAdapter/                      <- repo root (train.py lives here)
  .claude/skills/loradapter-mentor/
    SKILL.md
    references.md
  CLAUDE.md                      <- create this (template below)
```
Create .claude/skills/ if missing; Claude Code auto-discovers skills there.

## B. Verify the skill loaded
Ask Claude Code: "What skills do you have access to?" — you should see
loradapter-mentor. If not, restart the session (skills load at startup).

## C. CLAUDE.md (project memory — paste into repo root)
```
# Project: LoRAdapter (CTRLorALTer) — beginner mentee

- I have ZERO diffusion background. Teach concepts from scratch first.
- Source of truth: this repo's code + arXiv:2405.07913 + the loradapter-mentor
  skill's references.md. NEVER invent file names/APIs — read the file or search
  the web and verify.
- Do NOT rewrite my working scripts. Read them, propose the smallest change,
  explain WHY before changing.
- Comment code heavily — I am learning from it.
- Keep the project structure in references.md section 6. One concern per file.
  Calculation scripts never contain training logic. Depth and segmentation
  never share code paths. All generated artifacts under outputs/.
- I train on PRE-SAVED maps but infer with LIVE maps; saved and live maps MUST
  use the same model + preprocessing + value range. Verify the repo's midas
  encoder before writing any depth preprocessing.
- ENVIRONMENT: conda env `loradapter`. In a fresh PowerShell, before any
  python/train/sample command, run:
  (D:\MyWorkplace\installedSW\miniforge3\shell\condabin\conda-hook.ps1) ; (conda activate loradapter)
```

## D. DO THIS FIRST — resolve the open questions before writing pipeline code
Make a safety branch: git checkout -b mentor-review

Then paste into Claude Code:
"Read references.md. Answer the OPEN QUESTIONS in section 4 by reading the
 actual files: find the repo's MiDaS encoder (the module behind
 lora/encoder@lora.struct.encoder: midas — search configs/lora/encoder and src
 for midas), and read how forward() in src/model.py calls encoder() and uses
 skip_encode. Quote exact lines. Fill in the VERIFIED ANSWER blanks. Do NOT
 write any new pipeline code yet."

Why first: your plan (train on saved maps, infer live, keep them 'similar')
only works if your offline depth script reproduces the repo's live MiDaS
EXACTLY. We cannot write that script correctly until we've read which MiDaS
variant + preprocessing the repo uses. This step prevents wasted effort.

## E. Then audit your existing scripts (don't regenerate from scratch)
"List every script I've already written. For each: (a) what it does, (b) is it
 correct given the verified repo mechanics in references.md, (c) the single most
 important fix. Output an audit table. Do NOT rewrite anything yet."

## F. Build order (depth fully, THEN segmentation)
1. Stage A: calculate_depth_map.py (matches repo MiDaS exactly) -> outputs/depth_maps/
2. build_jsonl.py -> records {image_path, depth_map_path, prompt}
3. Stage B: local_depth.py dataset + train_depth.yaml + forward wiring for saved maps
4. Add the 4-image grid + prompt .txt + logs + tensorboard to the val block
5. Train depth, inspect grids. Only then mirror into Stage C/D for segmentation.

## G. Conda env: make activation automatic (optional)
notepad $PROFILE  -> add:
  (D:\MyWorkplace\installedSW\miniforge3\shell\condabin\conda-hook.ps1)
  conda activate loradapter
Then new terminals auto-activate; tell Claude Code "env auto-activates, skip
the hook line." (Verify the .ps1 path actually exists on your machine.)

## H. Quick sanity facts already verified from the repo (so you're grounded)
- Stock training data = (image, prompt) only; depth is computed LIVE by the
  midas encoder inside model.forward(). You are deliberately changing this to
  use saved maps.
- Training/sample size is 512x512; the paper center-crops to 512. Your
  1280x800 images need a deliberate crop-vs-letterbox choice (references.md §5).
- Base model is SD1.5 (train_struct_sd15.yaml). Validation already logs images
  + loss/lr to tensorboard — we EXTEND that block, not replace it.
