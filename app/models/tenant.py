"""app/models/tenant.py — Tenant profile ORM model."""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

if TYPE_CHECKING:
    from app.models.call import Call


class Tenant(Base):
    """Extracted tenant screening data from a completed call."""

    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=True,
        unique=True,
    )
    phone_number: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    # Extracted answers
    full_name: Mapped[str | None] = mapped_column(String(255))
    contact_phone: Mapped[str | None] = mapped_column(String(20))
    email: Mapped[str | None] = mapped_column(String(255))
    adults_count: Mapped[int | None] = mapped_column(Integer)
    children_count: Mapped[int | None] = mapped_column(Integer)
    occupants_count: Mapped[int | None] = mapped_column(Integer)
    monthly_income: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    income_raw: Mapped[str | None] = mapped_column(String(255))
    has_pets: Mapped[bool | None] = mapped_column(Boolean)
    pets_raw: Mapped[str | None] = mapped_column(Text)
    pet_type: Mapped[str | None] = mapped_column(String(100))
    pet_breed: Mapped[str | None] = mapped_column(String(100))
    pet_weight: Mapped[int | None] = mapped_column(Integer)
    has_eviction: Mapped[bool | None] = mapped_column(Boolean)
    eviction_raw: Mapped[str | None] = mapped_column(Text)
    eviction_circumstances: Mapped[str | None] = mapped_column(Text)
    move_in_date: Mapped[date | None] = mapped_column(Date)
    move_in_raw: Mapped[str | None] = mapped_column(String(255))
    current_residence: Mapped[str | None] = mapped_column(String(500))
    residence_duration: Mapped[str | None] = mapped_column(String(255))
    move_reason: Mapped[str | None] = mapped_column(Text)
    move_timing: Mapped[str | None] = mapped_column(String(255))
    employer: Mapped[str | None] = mapped_column(String(255))
    employment_duration: Mapped[str | None] = mapped_column(String(255))
    general_notes: Mapped[str | None] = mapped_column(Text)
    special_notes: Mapped[str | None] = mapped_column(Text)
    human_requested: Mapped[bool | None] = mapped_column(Boolean)
    callback_requested: Mapped[bool | None] = mapped_column(Boolean)
    stop_requested: Mapped[bool | None] = mapped_column(Boolean)
    raw_answers: Mapped[dict | None] = mapped_column(JSONB)
    normalized_data: Mapped[dict | None] = mapped_column(JSONB)
    answered_states: Mapped[list | None] = mapped_column(JSONB)
    refused_states: Mapped[list | None] = mapped_column(JSONB)
    faq_topics: Mapped[list | None] = mapped_column(JSONB)
    control_flags: Mapped[dict | None] = mapped_column(JSONB)
    qualification_details: Mapped[dict | None] = mapped_column(JSONB)

    # Qualification
    qualification_score: Mapped[int | None] = mapped_column(Integer)
    qualification_status: Mapped[str | None] = mapped_column(String(20), index=True)
    disqualify_reasons: Mapped[list | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)

    # Meta
    email_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    email_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reviewed_by_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_blacklisted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationship
    call: Mapped[Optional["Call"]] = relationship("Call", back_populates="tenant")

    def __repr__(self) -> str:
        return f"<Tenant {self.phone_number} | {self.qualification_status} | score={self.qualification_score}>"
