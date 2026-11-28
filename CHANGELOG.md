# Changelog

## 0.2.0 (2026-11-28)

From a finished dataset to a trained, scored LoRA, hands off.

### Training and scoring
- Create the AI-Toolkit job from a finished dataset, launch the training from Dataset
  Studio and follow it; copy the LoRA into an engine's LoRA folder.
- Score LoRA: the likeness of the trained LoRA on the card scale, with and without it;
  keep only the best save in the engine.
- Presets beyond Klein: FLUX.3, Z-Image Turbo, Qwen-Image, FLUX.1 Krea dev, Krea 2
  Turbo, Ideogram 4, MOSS-Image SFT and Base, MiniMax-H3-Image, and an SDXL checkpoint
  (Illustrious, Pony) with tag captions and Re-caption.
- Train on your own checkpoint for every preset ai-toolkit can load one for (Ideogram 4
  refused by name, Klein and Krea 2 bf16 only); the score uses the same file when the
  engine has it.
- Send to training: the rest of the chain on a finished dataset, by hand.

### Autopilot and queue
- One click from the reference photo to a scored LoRA; a queue of projects, one after
  the other, with the engine swapped between steps.
- The queue follows each project's "Run goes all the way to the LoRA" box (dataset
  only otherwise) and shows it before the start.
- An interrupted chain says so and can be resumed with its queue.
- Safeguards for unattended runs: gated Hugging Face check before the first card,
  trigger word in the LoRA metadata, desktop notifications.
- A text-to-image dataset engine (crispz-ideogram, crispz-krea2) is refused before the
  queue starts.

### Runs
- Stop after 3 tries in a row copied back from the reference (setting "stop after"),
  and after 3 cards in a row failing with the same engine error.
- The edit speed mode (Lightning) goes to qwen-edit only.
- An engine still booting is waited for; localhost is always in NO_PROXY (Gradio quit
  at boot behind a proxy).

### Projects and review
- Clone a project (with or without its renders), Delete a project.
- Review tools: cards too close to each other, Info, Compare; send a photo from
  IrfanView.

### Interface
- Settings: the dialog scrolls; nameless engines and shared ports are refused.
- A loading spinner at start.
- Terminal: a start banner (engines, ai-toolkit, projects) and coloured log lines.

## 0.1.0 (2026-10-29)

First version: a standalone app that drives and launches the crispz family engines,
renders a captioned character dataset from one reference photo and scores each card's
likeness against it (ArcFace).
