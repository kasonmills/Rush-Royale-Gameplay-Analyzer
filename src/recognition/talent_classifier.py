"""
Talent icon classifier — reads the talent badge from each board cell and
identifies which tier (1, 2, 3, or 4) and which branch (L, R, Fixed) it
represents. Runs as a parallel task independent of unit body classification.

Recognition strategy:
  - Template match the badge region against reference icons for all tiers
    and branches (loaded from the visual reference library)
  - Returns (tier, branch, confidence) or None if no talent badge is visible

Phase 1 implementation.
"""
# TODO (Phase 1): implement talent icon template matching against all tier/branch icons
