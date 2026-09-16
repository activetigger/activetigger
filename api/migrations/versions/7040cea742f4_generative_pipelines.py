"""Generative pipelines (issue #1100)

Replaces the per-project gen_models + per-answer generations + prompts
tables with:
- gen_credentials: endpoint/key pairs, user or instance level
- gen_pipelines: credentials + model + parameters + prompt + post-treatment
- generations: one row per run of a pipeline (outputs stored in files)

Data from the old tables is NOT migrated.

Revision ID: 7040cea742f4
Revises: e9c6a551b6fa
Create Date: 2026-09-14

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7040cea742f4"
down_revision: Union[str, None] = "e9c6a551b6fa"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("generations")
    op.drop_table("gen_models")
    op.drop_table("prompts")

    op.create_table(
        "gen_credentials",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), nullable=False),
        sa.Column("api_key", sa.String(), nullable=False),
        sa.Column("models", sa.JSON(), nullable=True),
        sa.Column("last_tested", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_name"],
            ["users.user_name"],
            name=op.f("fk_gen_credentials_user_name_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_gen_credentials")),
        sa.UniqueConstraint("user_name", "name", name="uq_gen_credentials_user_name_name"),
    )

    op.create_table(
        "gen_pipelines",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("project_slug", sa.String(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("scheme_name", sa.String(), nullable=True),
        sa.Column("credentials_id", sa.Integer(), nullable=False),
        sa.Column("model_slug", sa.String(), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("postprocess", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_slug"],
            ["projects.project_slug"],
            name=op.f("fk_gen_pipelines_project_slug_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_name"],
            ["users.user_name"],
            name=op.f("fk_gen_pipelines_user_name_users"),
        ),
        sa.ForeignKeyConstraint(
            ["credentials_id"],
            ["gen_credentials.id"],
            name=op.f("fk_gen_pipelines_credentials_id_gen_credentials"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_gen_pipelines")),
        sa.UniqueConstraint("project_slug", "name", name="uq_gen_pipelines_project_slug_name"),
    )

    op.create_table(
        "generations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("pipeline_id", sa.Integer(), nullable=False),
        sa.Column("user_name", sa.String(), nullable=False),
        sa.Column("project_slug", sa.String(), nullable=False),
        sa.Column("dataset", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("n_elements", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["pipeline_id"],
            ["gen_pipelines.id"],
            name=op.f("fk_generations_pipeline_id_gen_pipelines"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_name"],
            ["users.user_name"],
            name=op.f("fk_generations_user_name_users"),
        ),
        sa.ForeignKeyConstraint(
            ["project_slug"],
            ["projects.project_slug"],
            name=op.f("fk_generations_project_slug_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_generations")),
    )


def downgrade() -> None:
    # the old tables are not restored with their data; recreate the new ones
    # from a database backup if needed
    op.drop_table("generations")
    op.drop_table("gen_pipelines")
    op.drop_table("gen_credentials")
