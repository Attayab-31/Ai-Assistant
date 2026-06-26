"""
app/models/call.py — Call records ORM model.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.tenant import Tenant


class Call(Base):
    """Represents a single phone call screening session."""

    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    call_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    phone_number: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(10), default="inbound")
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="initiated", index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    full_transcript: Mapped[str | None] = mapped_column(Text)
    recording_url: Mapped[str | None] = mapped_column(Text)
    questions_answered: Mapped[int] = mapped_column(Integer, default=0)
    call_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    stt_provider: Mapped[str | None] = mapped_column(String(50))
    llm_provider: Mapped[str | None] = mapped_column(String(50))
    tts_provider: Mapped[str | None] = mapped_column(String(50))
    error_log: Mapped[dict | None] = mapped_column(JSONB)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationship
    tenant: Mapped[Optional["Tenant"]] = relationship(
        "Tenant", back_populates="call", uselist=False
    )

    def __repr__(self) -> str:
        return f"<Call {self.call_id} | {self.phone_number} | {self.status}>"
