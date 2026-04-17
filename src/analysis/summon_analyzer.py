"""
SummonAnalyzer — statistical analysis of Rush Royale summon randomness.

Hypothesis being tested
-----------------------
The game claims each unit in a 5-unit deck has a 20% chance of being summoned
whenever a new unit is generated.  This module tests whether the observed
distribution is consistent with a uniform 20% for each unit, or whether
hidden weights exist.

Two contexts are analysed separately:
  'manual'     — summons triggered by the player spending mana
  'post_merge' — summons triggered automatically after a merge frees a cell

Statistical tests
-----------------
1. Chi-squared goodness-of-fit — overall test for the full deck distribution.
   H₀: p_i = 1/k for all i (uniform across k units)
   A significant p-value (< 0.05) means the distribution is unlikely uniform.

2. Per-unit two-tailed Z-test for proportions.
   H₀: p_i = 1/k  (this unit's true rate equals 1/k)
   Flags individual units that are drawing disproportionately often or rarely.

3. Wilson score 95% confidence interval per unit.
   Gives the plausible range for each unit's true spawn probability.

Reliability note
----------------
Chi-squared requires at least ~5 expected observations per cell for valid
results.  With k=5 and expected_per_unit = n/5, you need n ≥ 25 at minimum,
but n ≥ 100 is recommended for reliable p-values.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

try:
    from scipy.stats import chi2 as _chi2_dist, norm as _norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class UnitStats:
    unit_id:        str
    display_name:   str          # filled in by caller from unit_meta.db
    observed:       int
    expected:       float        # n / deck_size
    observed_rate:  float        # observed / total
    expected_rate:  float        # 1 / deck_size
    z_score:        float        # (observed_rate - expected_rate) / se
    p_value:        float        # two-tailed p-value for this unit's Z-test
    ci_low:         float        # Wilson 95% CI lower bound
    ci_high:        float        # Wilson 95% CI upper bound
    flagged:        bool         # True if p_value < 0.05

    @property
    def deviation_pct(self) -> float:
        """How far the observed rate deviates from expected, as a percentage."""
        return (self.observed_rate - self.expected_rate) * 100.0


@dataclass
class MergeStats:
    unit_id:      str
    display_name: str
    from_rank:    int
    to_rank:      int
    count:        int


@dataclass
class SummonAnalysisResult:
    # Context this result covers
    trigger_type:   str          # 'all', 'manual', 'post_merge'
    total_summons:  int
    deck_size:      int
    deck_units:     list[str]    # unit_ids (sorted)

    # Per-unit breakdown
    unit_stats:     list[UnitStats] = field(default_factory=list)

    # Overall chi-squared test
    chi_sq_statistic: float = 0.0
    chi_sq_df:        int   = 0
    chi_sq_p_value:   float = 1.0

    # Human-readable verdict
    verdict:      str = "Insufficient data"
    verdict_detail: str = ""

    # Merge stats (populated separately, not split by trigger_type)
    merge_stats:  list[MergeStats] = field(default_factory=list)

    # Minimum recommended sample size for reliable results
    MIN_RELIABLE = 100

    @property
    def reliable(self) -> bool:
        return self.total_summons >= self.MIN_RELIABLE

    @property
    def expected_per_unit(self) -> float:
        if self.deck_size == 0:
            return 0.0
        return self.total_summons / self.deck_size


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _chi2_p_value(statistic: float, df: int) -> float:
    """Return the survival function (1 - CDF) of the chi-squared distribution."""
    if not _HAS_SCIPY:
        # Fallback: regularised upper incomplete gamma approximation.
        # Less accurate for small df but sufficient for a flag/notice.
        return _chi2_sf_approx(statistic, df)
    return float(_chi2_dist.sf(statistic, df))


def _normal_p_value(z: float) -> float:
    """Two-tailed p-value for a standard normal Z score."""
    if not _HAS_SCIPY:
        return 2.0 * (1.0 - _normal_cdf_approx(abs(z)))
    return float(2.0 * _norm.sf(abs(z)))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for proportion k/n at given z."""
    if n == 0:
        return 0.0, 1.0
    p_hat = k / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# Numerical approximations used when scipy is absent ---

def _normal_cdf_approx(x: float) -> float:
    """Abramowitz & Stegun approximation for standard normal CDF (error < 7.5e-8)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    pdf  = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    cdf  = 1.0 - pdf * poly
    return cdf if x >= 0 else 1.0 - cdf


def _chi2_sf_approx(x: float, df: int) -> float:
    """
    Approximate chi-squared survival function via the Wilson–Hilferty
    cube-root normal transformation.  Accurate to ~2 decimal places in p.
    """
    if x <= 0:
        return 1.0
    k = df / 2.0
    mu  = 1.0 - 2.0 / (9 * df)
    sig = math.sqrt(2.0 / (9 * df))
    z   = ((x / df) ** (1.0 / 3.0) - mu) / sig
    return 1.0 - _normal_cdf_approx(z)


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class SummonAnalyzer:
    """
    Computes summon and merge statistics from summon_analysis.db.

    Usage::

        from src.database.connection import summon_analysis_db
        from src.analysis.summon_analyzer import SummonAnalyzer

        with summon_analysis_db() as conn:
            result = SummonAnalyzer.analyse(conn, trigger_type='all')
    """

    @staticmethod
    def analyse(
        sa_conn: sqlite3.Connection,
        trigger_type: str = "all",
        unit_names: Optional[dict[str, str]] = None,
    ) -> SummonAnalysisResult:
        """
        Run the full statistical analysis and return a SummonAnalysisResult.

        Args:
            sa_conn:      Open connection to summon_analysis.db.
            trigger_type: 'all', 'manual', or 'post_merge'.
            unit_names:   Optional unit_id → display_name mapping for labels.
        """
        unit_names = unit_names or {}
        tt = None if trigger_type == "all" else trigger_type

        # ---- Fetch summon counts per unit ----
        rows = sa_conn.execute(
            """SELECT unit_summoned, COUNT(*) AS count
               FROM summon_events e
               JOIN summon_sessions s USING (match_id)
               WHERE s.deck_json IS NOT NULL
               """ + ("AND e.trigger_type = ?" if tt else "") + """
               GROUP BY unit_summoned""",
            (tt,) if tt else (),
        ).fetchall()

        total = sum(r["count"] for r in rows)
        counts: dict[str, int] = {r["unit_summoned"]: r["count"] for r in rows}

        # ---- Determine deck units from completed sessions ----
        deck_rows = sa_conn.execute(
            "SELECT deck_json FROM summon_sessions WHERE deck_json IS NOT NULL"
        ).fetchall()

        deck_units: set[str] = set()
        for dr in deck_rows:
            try:
                deck_units.update(json.loads(dr["deck_json"]))
            except (json.JSONDecodeError, TypeError):
                pass
        deck_units.update(counts.keys())   # also include any unit we saw
        deck_units_sorted = sorted(deck_units)
        deck_size = len(deck_units_sorted)

        result = SummonAnalysisResult(
            trigger_type=trigger_type,
            total_summons=total,
            deck_size=deck_size,
            deck_units=deck_units_sorted,
        )

        if total == 0 or deck_size == 0:
            result.verdict = "No data yet"
            result.verdict_detail = "Start a match to begin collecting summon events."
            return result

        # ---- Per-unit stats ----
        expected_rate = 1.0 / deck_size
        expected_n    = total * expected_rate
        chi_sq        = 0.0

        for uid in deck_units_sorted:
            obs   = counts.get(uid, 0)
            obs_r = obs / total
            se    = math.sqrt(expected_rate * (1 - expected_rate) / total) if total > 0 else 1.0
            z     = (obs_r - expected_rate) / se if se > 0 else 0.0
            p_val = _normal_p_value(z)
            ci_lo, ci_hi = _wilson_ci(obs, total)
            chi_sq += (obs - expected_n) ** 2 / expected_n

            result.unit_stats.append(UnitStats(
                unit_id       = uid,
                display_name  = unit_names.get(uid, uid),
                observed      = obs,
                expected      = round(expected_n, 1),
                observed_rate = round(obs_r, 4),
                expected_rate = round(expected_rate, 4),
                z_score       = round(z, 3),
                p_value       = round(p_val, 4),
                ci_low        = round(ci_lo, 4),
                ci_high       = round(ci_hi, 4),
                flagged       = p_val < 0.05,
            ))

        # ---- Overall chi-squared ----
        df    = deck_size - 1
        p_chi = _chi2_p_value(chi_sq, df)
        result.chi_sq_statistic = round(chi_sq, 3)
        result.chi_sq_df        = df
        result.chi_sq_p_value   = round(p_chi, 4)

        # ---- Verdict ----
        if not result.reliable:
            result.verdict = "Collecting data…"
            result.verdict_detail = (
                f"{total} summons recorded — {result.MIN_RELIABLE} needed for reliable results."
            )
        elif p_chi < 0.01:
            result.verdict = "Significant Bias Detected"
            flagged = [u.display_name for u in result.unit_stats if u.flagged]
            result.verdict_detail = (
                f"p = {p_chi:.4f} — extremely unlikely if truly uniform. "
                f"Flagged units: {', '.join(flagged) or 'none individually significant'}."
            )
        elif p_chi < 0.05:
            result.verdict = "Suspicious Distribution"
            result.verdict_detail = (
                f"p = {p_chi:.4f} — distribution is unlikely uniform at 95% confidence."
            )
        else:
            result.verdict = "Consistent with Fair Randomness"
            result.verdict_detail = (
                f"p = {p_chi:.4f} — no statistically significant deviation from 20% per unit."
            )

        # ---- Merge stats ----
        merge_rows = sa_conn.execute(
            """SELECT m.unit_id, m.from_rank, m.to_rank, COUNT(*) AS count
               FROM merge_events m
               JOIN summon_sessions s USING (match_id)
               WHERE s.deck_json IS NOT NULL
               GROUP BY m.unit_id, m.from_rank, m.to_rank
               ORDER BY m.unit_id, m.from_rank""",
        ).fetchall()

        for mr in merge_rows:
            result.merge_stats.append(MergeStats(
                unit_id      = mr["unit_id"],
                display_name = unit_names.get(mr["unit_id"], mr["unit_id"]),
                from_rank    = mr["from_rank"],
                to_rank      = mr["to_rank"],
                count        = mr["count"],
            ))

        return result
