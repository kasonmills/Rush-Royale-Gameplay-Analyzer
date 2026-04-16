"""
Repository classes for unit_meta.db.

Each class wraps one or more related tables and exposes typed read/write
methods. All methods accept a sqlite3.Connection — callers manage the
connection lifetime via the context managers in connection.py.

Example:
    from src.database.connection import unit_meta_db
    from src.database.unit_meta_repo import UnitRepo, TalentRepo, HeroRepo, TierScoreRepo

    with unit_meta_db() as conn:
        unit = UnitRepo.get(conn, "inquisitor")
        talents = TalentRepo.get_for_unit(conn, "inquisitor")
"""

import sqlite3


# ---------------------------------------------------------------------------
# UnitRepo — units table
# ---------------------------------------------------------------------------

class UnitRepo:

    @staticmethod
    def get(conn: sqlite3.Connection, unit_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM units WHERE unit_id = ?", (unit_id,)
        ).fetchone()

    @staticmethod
    def get_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute("SELECT * FROM units ORDER BY display_name").fetchall()

    @staticmethod
    def get_by_role(conn: sqlite3.Connection, role: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM units WHERE primary_role = ? OR secondary_role = ? ORDER BY display_name",
            (role, role)
        ).fetchall()

    @staticmethod
    def upsert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        updates = ", ".join(f"{k} = excluded.{k}" for k in data if k != "unit_id")
        conn.execute(
            f"INSERT INTO units ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(unit_id) DO UPDATE SET {updates}",
            list(data.values())
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for row in rows:
            UnitRepo.upsert(conn, row)


# ---------------------------------------------------------------------------
# TalentRepo — talent_trees table
# ---------------------------------------------------------------------------

class TalentRepo:

    @staticmethod
    def get_for_unit(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM talent_trees WHERE unit_id = ? ORDER BY tier, branch",
            (unit_id,)
        ).fetchall()

    @staticmethod
    def get_branch(conn: sqlite3.Connection, unit_id: str, tier: int, branch: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM talent_trees WHERE unit_id = ? AND tier = ? AND branch = ?",
            (unit_id, tier, branch)
        ).fetchone()

    @staticmethod
    def upsert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        conn.execute(
            f"INSERT INTO talent_trees ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(unit_id, tier, branch) DO UPDATE SET "
            f"talent_name = excluded.talent_name, "
            f"mechanical_effect = excluded.mechanical_effect, "
            f"observable_sigs = excluded.observable_sigs, "
            f"research_status = excluded.research_status, "
            f"last_updated = excluded.last_updated",
            list(data.values())
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for row in rows:
            TalentRepo.upsert(conn, row)


# ---------------------------------------------------------------------------
# StatNumberRepo — stat_numbers table
# ---------------------------------------------------------------------------

class StatNumberRepo:

    @staticmethod
    def get_for_unit(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM stat_numbers WHERE unit_id = ?", (unit_id,)
        ).fetchall()

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        if not rows:
            return
        unit_ids = list({r["unit_id"] for r in rows if r.get("unit_id")})
        if unit_ids:
            placeholders = ",".join("?" * len(unit_ids))
            conn.execute(f"DELETE FROM stat_numbers WHERE unit_id IN ({placeholders})", unit_ids)
        for data in rows:
            cols = ", ".join(data.keys())
            ph = ", ".join("?" * len(data))
            conn.execute(f"INSERT INTO stat_numbers ({cols}) VALUES ({ph})", list(data.values()))


# ---------------------------------------------------------------------------
# AnimationRepo — animations table
# ---------------------------------------------------------------------------

class AnimationRepo:

    @staticmethod
    def get_for_unit(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM animations WHERE unit_id = ? ORDER BY animation_name",
            (unit_id,)
        ).fetchall()

    @staticmethod
    def get_by_category(conn: sqlite3.Connection, category: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM animations WHERE category = ? ORDER BY unit_id",
            (category,)
        ).fetchall()

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        if not rows:
            return
        unit_ids = list({r["unit_id"] for r in rows if r.get("unit_id")})
        if unit_ids:
            placeholders = ",".join("?" * len(unit_ids))
            conn.execute(f"DELETE FROM animations WHERE unit_id IN ({placeholders})", unit_ids)
        for data in rows:
            cols = ", ".join(data.keys())
            ph = ", ".join("?" * len(data))
            conn.execute(f"INSERT INTO animations ({cols}) VALUES ({ph})", list(data.values()))


# ---------------------------------------------------------------------------
# SynergyRepo — synergies table
# ---------------------------------------------------------------------------

class SynergyRepo:

    @staticmethod
    def get_for_unit(conn: sqlite3.Connection, unit_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM synergies WHERE unit_a_id = ? OR unit_b_id = ?",
            (unit_id, unit_id)
        ).fetchall()

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        if not rows:
            return
        unit_a_ids = list({r["unit_a_id"] for r in rows if r.get("unit_a_id")})
        if unit_a_ids:
            placeholders = ",".join("?" * len(unit_a_ids))
            conn.execute(f"DELETE FROM synergies WHERE unit_a_id IN ({placeholders})", unit_a_ids)
        for data in rows:
            cols = ", ".join(data.keys())
            ph = ", ".join("?" * len(data))
            conn.execute(f"INSERT INTO synergies ({cols}) VALUES ({ph})", list(data.values()))


# ---------------------------------------------------------------------------
# HeroRepo — heroes, hero_ability_sets, hero_investment_sets tables
# ---------------------------------------------------------------------------

class HeroRepo:

    @staticmethod
    def get_ability_sets(conn: sqlite3.Connection, hero_id: str) -> list[sqlite3.Row]:
        """Returns all ability sets for a hero ordered by set number."""
        return conn.execute(
            "SELECT * FROM hero_ability_sets WHERE hero_id = ? ORDER BY set_number",
            (hero_id,)
        ).fetchall()

    @staticmethod
    def get_investable_stats(conn: sqlite3.Connection, hero_id: str,
                             set_number: int) -> list[sqlite3.Row]:
        """Returns all investable stat categories for one ability set."""
        return conn.execute(
            "SELECT * FROM hero_investment_sets WHERE hero_id = ? AND set_number = ?",
            (hero_id, set_number)
        ).fetchall()

    @staticmethod
    def get_all_heroes(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute("SELECT * FROM heroes ORDER BY hero_id").fetchall()

    @staticmethod
    def upsert_hero(conn: sqlite3.Connection, hero_id: str, display_name: str):
        conn.execute(
            "INSERT INTO heroes (hero_id, display_name) VALUES (?, ?) "
            "ON CONFLICT(hero_id) DO UPDATE SET display_name = excluded.display_name",
            (hero_id, display_name)
        )

    @staticmethod
    def upsert_ability_set(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        updates = ", ".join(f"{k} = excluded.{k}" for k in data
                            if k not in ("hero_id", "set_number"))
        conn.execute(
            f"INSERT INTO hero_ability_sets ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(hero_id, set_number) DO UPDATE SET {updates}",
            list(data.values())
        )

    @staticmethod
    def upsert_investment_sets(conn: sqlite3.Connection, hero_id: str,
                               set_number: int, investment_sets: list[dict]):
        """Replaces all investment sets for a given hero/ability-set combination."""
        conn.execute(
            "DELETE FROM hero_investment_sets WHERE hero_id = ? AND set_number = ?",
            (hero_id, set_number)
        )
        for inv in investment_sets:
            inv_data = {"hero_id": hero_id, "set_number": set_number, **inv}
            cols = ", ".join(inv_data.keys())
            placeholders = ", ".join("?" * len(inv_data))
            conn.execute(
                f"INSERT INTO hero_investment_sets ({cols}) VALUES ({placeholders})",
                list(inv_data.values())
            )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        """
        Expects rows in the format produced by sync_from_sheet:
          {hero_id, display_name, set_number, ability_name, ability_type,
           morale_cost, description, unlock_points_required, observable_sigs,
           research_status, last_updated,
           investment_sets: [{investment_name, total_point_limit, requires_80_pts,
                              pre_80_point_limit, description}]}
        """
        for row in rows:
            investment_sets = row.pop("investment_sets", [])
            HeroRepo.upsert_hero(conn, row["hero_id"], row["display_name"])
            ability_set = {k: v for k, v in row.items() if k != "display_name"}
            HeroRepo.upsert_ability_set(conn, ability_set)
            if investment_sets:
                HeroRepo.upsert_investment_sets(
                    conn, row["hero_id"], row["set_number"], investment_sets
                )


# ---------------------------------------------------------------------------
# TierScoreRepo — tier_scores table
# ---------------------------------------------------------------------------

class TierScoreRepo:

    @staticmethod
    def get_for_entity(conn: sqlite3.Connection, entity_id: str) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM tier_scores WHERE entity_id = ? ORDER BY entity_build",
            (entity_id,)
        ).fetchall()

    @staticmethod
    def get_by_tier(conn: sqlite3.Connection, tier: str, entity_type: str = "Unit") -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM tier_scores WHERE tier = ? AND entity_type = ? ORDER BY score DESC",
            (tier, entity_type)
        ).fetchall()

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        if not rows:
            return
        entity_ids = list({r["entity_id"] for r in rows if r.get("entity_id")})
        if entity_ids:
            placeholders = ",".join("?" * len(entity_ids))
            conn.execute(f"DELETE FROM tier_scores WHERE entity_id IN ({placeholders})", entity_ids)
        for data in rows:
            cols = ", ".join(data.keys())
            ph = ", ".join("?" * len(data))
            conn.execute(f"INSERT INTO tier_scores ({cols}) VALUES ({ph})", list(data.values()))


# ---------------------------------------------------------------------------
# PatchLogRepo — patch_log table
# ---------------------------------------------------------------------------

class PatchLogRepo:

    @staticmethod
    def get(conn: sqlite3.Connection, patch_version: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM patch_log WHERE patch_version = ?", (patch_version,)
        ).fetchone()

    @staticmethod
    def get_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute(
            "SELECT * FROM patch_log ORDER BY release_date DESC"
        ).fetchall()

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for data in rows:
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO patch_log ({cols}) VALUES ({placeholders}) "
                f"ON CONFLICT(patch_version) DO UPDATE SET "
                f"release_date = excluded.release_date, "
                f"units_changed = excluded.units_changed, "
                f"heroes_changed = excluded.heroes_changed, "
                f"new_content = excluded.new_content, "
                f"notes = excluded.notes",
                list(data.values())
            )


# ---------------------------------------------------------------------------
# ArtifactRepo — artifacts table
# ---------------------------------------------------------------------------

class ArtifactRepo:

    @staticmethod
    def get(conn: sqlite3.Connection, artifact_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()

    @staticmethod
    def get_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute("SELECT * FROM artifacts ORDER BY display_name").fetchall()

    @staticmethod
    def upsert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        updates = ", ".join(f"{k} = excluded.{k}" for k in data if k != "artifact_id")
        conn.execute(
            f"INSERT INTO artifacts ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(artifact_id) DO UPDATE SET {updates}",
            list(data.values())
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for row in rows:
            ArtifactRepo.upsert(conn, row)


# ---------------------------------------------------------------------------
# SpellRepo — spells table
# ---------------------------------------------------------------------------

class SpellRepo:

    @staticmethod
    def get(conn: sqlite3.Connection, spell_id: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM spells WHERE spell_id = ?", (spell_id,)
        ).fetchone()

    @staticmethod
    def get_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
        return conn.execute("SELECT * FROM spells ORDER BY display_name").fetchall()

    @staticmethod
    def upsert(conn: sqlite3.Connection, data: dict):
        cols = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        updates = ", ".join(f"{k} = excluded.{k}" for k in data if k != "spell_id")
        conn.execute(
            f"INSERT INTO spells ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(spell_id) DO UPDATE SET {updates}",
            list(data.values())
        )

    @staticmethod
    def upsert_many(conn: sqlite3.Connection, rows: list[dict]):
        for row in rows:
            SpellRepo.upsert(conn, row)
