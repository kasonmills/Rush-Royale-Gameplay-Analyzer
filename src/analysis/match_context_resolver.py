"""
Match Context Resolver (MCR) — 5-stage cascade that identifies both decks,
narrows the active reference set, and tracks recognition confidence per unit
throughout a match. Phase 2 implementation.

Stages:
  1. Initial board scan — detect all visible units via template matching
  2. Deck hypothesis formation — infer the 5-unit deck per player
  3. Reference set narrowing — reduce active references to matched deck units
  4. Confidence tracking — update per-unit classification confidence each frame
  5. Special-case handling — trigger class-aware logic for Twins, Inquisitor, etc.
"""
# TODO (Phase 2): implement MCR cascade
