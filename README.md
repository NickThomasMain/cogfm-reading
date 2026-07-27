# cogfm-reading

Cognitive Foundation Model for Reading — binding heterogeneous cognitive
signals (Eye-Tracking, EEG, optionally fMRI) to a **language-anchored**
semantic space, in the spirit of ImageBind/LanguageBind but with a frozen
LLM (Qwen3) as the anchor instead of vision.

Master's thesis project. Methods-and-engineering research on public,
consent-based, anonymised datasets. This repository is the implementation;
the thesis text lives separately in the Obsidian vault.

## Status

Milestone **M0 — skeleton**. Repository scaffold and Git foundation in place.
End-to-end smoke run, data adapters and feature cache follow (see roadmap).

## Idea in one paragraph

Modality encoders (LaBraM for EEG, a scanpath-only encoder for ET,
Brain-JEPA for fMRI) and the LLM anchor are **frozen**. Only a lightweight
**connector** and a **PEFT** adapter are trained. Phase 1 binds each modality
to the anchor's word/sub-word embeddings with a composite SigLIP +
regression loss. Phase 2 adapts to downstream tasks via cognitive soft-prompts
+ LoRA. Because the backbones are frozen, encoder outputs are cached once and
all connector/loss ablations read from the cache — which makes the
ablation-heavy (OFAT) methodology affordable.

## Planned layout

```
configs/      Hydra configs — the single source for experiments (OFAT sweeps)
src/cogfm/
  data/       adapters (one per source) -> canonical sample; split-manager
  features/   feature cache keyed on (dataset, subject, prep-hash, encoder-id)
  encoders/   frozen modality encoders (shared base class)
  connectors/ Linear / MLP / QFormer / QConformer
  losses/     SigLIP, InfoNCE, regression, combinations
  anchor/     LLM anchor interface (Qwen3): word/sub-word embeddings
  binding/    Phase-1 training loop (reads cached features)
  adaptation/ Phase-2: soft-prompts + PEFT
  eval/       subject-CV protocol, retrieval (floor/ceiling), downstream, stats
  registry.py name -> class, for OFAT component swapping
tests/        adapter shapes, split disjointness, loss math, cache keys
scripts/      extract_features, run_binding, run_downstream, run_sweep
notebooks/    demonstration/analysis only — never a source of results
```

## Roadmap (milestones)

- **M0** Skeleton + end-to-end smoke run on dummy data
- **M1** Data adapters (ET, EEG) + feature cache + split-manager
- **M2** First real binding: ET -> text retrieval above chance
- **M3** EEG added; first emergence measurement (ET<->EEG on ZuCo) with floor + ceiling
- **M4** Phase 2 + downstream tasks + no-LLM / no-binding ablations
- **M5** OFAT sweeps + mandatory ablations
- **M6** fMRI (optional extension)
- **M7** Consolidation, reproducibility, writing

## License

MIT — see [LICENSE](LICENSE).
