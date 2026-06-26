"""app/models/settings.py — System settings ORM model."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class SystemSetting(Base):
    """Key-value store for all system settings, editable from admin panel."""

    __tablename__ = "system_settings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    key: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    value: Mapped[str] = mapped_column(Text, nullable=False)
    value_type: Mapped[str | None] = mapped_column(
        String(20)
    )  # string, json, boolean, integer
    description: Mapped[str | None] = mapped_column(Text)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("admin_users.id", ondelete="SET NULL"),
        nullable=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        display = "***" if self.is_sensitive else self.value[:50]
        return f"<SystemSetting {self.key}={display}>"
