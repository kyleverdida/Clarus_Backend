import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.database import get_user, get_user_for_auth
from app.user_model import User, UserRole

_TOKEN_ALGORITHM = "HS256"
_TOKEN_TTL_SECONDS = 60 * 60 * 8
_PASSWORD_SALT_BYTES = 16
_PASSWORD_KEY_BYTES = 32
_PASSWORD_SCRYPT_N = 2**14
_PASSWORD_SCRYPT_R = 8
_PASSWORD_SCRYPT_P = 1
_bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Passwords must be at least 8 characters long.")
    salt = secrets.token_bytes(_PASSWORD_SALT_BYTES)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_PASSWORD_SCRYPT_N,
        r=_PASSWORD_SCRYPT_R,
        p=_PASSWORD_SCRYPT_P,
        dklen=_PASSWORD_KEY_BYTES,
    )
    return "$".join(
        [
            "scrypt",
            str(_PASSWORD_SCRYPT_N),
            str(_PASSWORD_SCRYPT_R),
            str(_PASSWORD_SCRYPT_P),
            _encode(salt),
            _encode(derived_key),
        ]
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, n, r, p, salt, expected_key = encoded_hash.split("$")
        if algorithm != "scrypt":
            return False
        derived_key = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=_PASSWORD_KEY_BYTES,
        )
        return hmac.compare_digest(derived_key, _decode(expected_key))
    except (TypeError, ValueError):
        return False


def create_access_token(user: dict) -> str:
    now = int(time.time())
    payload = {
        "sub": user["user_id"],
        "role": user["role"],
        "iat": now,
        "exp": now + _TOKEN_TTL_SECONDS,
    }
    encoded_header = _encode_json({"alg": _TOKEN_ALGORITHM, "typ": "JWT"})
    encoded_payload = _encode_json(payload)
    unsigned_token = f"{encoded_header}.{encoded_payload}"
    signature = _sign(unsigned_token)
    return f"{unsigned_token}.{signature}"


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = _decode_access_token(credentials.credentials)
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    user = get_user(user_id)
    if user is None or not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is inactive or no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_roles(*allowed_roles: UserRole) -> Callable:
    def dependency(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return current_user

    return dependency


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if not current_user["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges are required.",
        )
    return current_user


def authenticate_user(email: str, password: str) -> dict | None:
    user = get_user_for_auth(email)
    if user is None or not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    user.pop("password_hash", None)
    user["is_admin"] = bool(user["is_admin"])
    user["is_active"] = bool(user["is_active"])
    return user


def _sign(value: str) -> str:
    secret = os.getenv("CLARUS_AUTH_SECRET")
    if not secret:
        raise RuntimeError(
            "CLARUS_AUTH_SECRET must be configured before issuing access tokens."
        )
    digest = hmac.new(
        secret.encode("utf-8"), value.encode("ascii"), hashlib.sha256
    ).digest()
    return _encode(digest)


def _decode_access_token(token: str) -> str:
    try:
        encoded_header, encoded_payload, received_signature = token.split(".")
        unsigned_token = f"{encoded_header}.{encoded_payload}"
        expected_signature = _sign(unsigned_token)
        if not hmac.compare_digest(received_signature, expected_signature):
            raise ValueError("Invalid token signature.")

        header = json.loads(_decode(encoded_header))
        payload = json.loads(_decode(encoded_payload))
        if header.get("alg") != _TOKEN_ALGORITHM:
            raise ValueError("Unsupported token algorithm.")
        if int(payload["exp"]) <= int(time.time()):
            raise ValueError("Token has expired.")
        return str(payload["sub"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        if isinstance(error, RuntimeError):
            raise
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _encode_json(value: dict) -> str:
    return _encode(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
