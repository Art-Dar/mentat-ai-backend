import enum
import uuid
from datetime import datetime

from sqlalchemy import CheckConstraint, ForeignKey, String, UniqueConstraint, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    email_verified: Mapped[bool] = mapped_column(default=False)
    display_name: Mapped[str | None] = mapped_column(String(255))
    identities: Mapped[list["AuthIdentity"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class AuthProvider(enum.StrEnum):
    PASSWORD = "password"
    GOOGLE = "google"


class AuthIdentity(Base):
    __tablename__ = "auth_identities"
    __table_args__ = (
        UniqueConstraint("provider", "provider_user_id"),
        # password_hash belongs to password only, and is required there
        CheckConstraint(
            "(provider = 'password' AND password_hash IS NOT NULL)"
            " OR (provider <> 'password' AND password_hash IS NULL)",
            name="ck_password_hash_only_for_password",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[AuthProvider] = mapped_column(
        SAEnum(
            AuthProvider,
            name="auth_provider",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        )
    )

    # Google's stable `sub` claim; for password, the normalized email
    provider_user_id: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str | None] = mapped_column(String(255))

    user: Mapped["User"] = relationship(back_populates="identities")