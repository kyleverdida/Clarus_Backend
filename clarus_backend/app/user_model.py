from dataclasses import dataclass
from datetime import datetime
from typing import Literal


UserRole = Literal[
    "health_worker",
    "specialist",
    "ophthalmologist_optometrist",
]

USER_ROLES: tuple[UserRole, ...] = (
    "health_worker",
    "specialist",
    "ophthalmologist_optometrist",
)


@dataclass(frozen=True)
class User:
    user_id: str
    full_name: str
    email: str
    role: UserRole
    is_admin: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: dict) -> "User":
        return cls(
            user_id=row["user_id"],
            full_name=row["full_name"],
            email=row["email"],
            role=row["role"],
            is_admin=bool(row["is_admin"]),
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def to_public_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "is_admin": self.is_admin,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
