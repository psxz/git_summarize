"""
app/core/security.py
─────────────────────
Authentication via Logto (OIDC / JWT) with a static-key fallback.

── Mode selection ────────────────────────────────────────────────────────────
Set LOGTO_ENDPOINT in the environment to enable Logto mode:

  Logto mode   — validates RS256 JWTs issued by your Logto tenant.
                 JWKS keys are fetched from {LOGTO_ENDPOINT}/oidc/jwks and
                 cached in-process (PyJWT's PyJWKClient handles rotation).
                 The validated JWT payload is returned as an AuthInfo object.

  Static mode  — falls back to the original shared-secret Bearer check when
                 LOGTO_ENDPOINT is not set (useful for dev / self-hosted).

── AuthInfo ──────────────────────────────────────────────────────────────────
A validated request yields an AuthInfo dataclass that downstream handlers
can depend on.  Key fields:

  sub               User identifier (Logto user ID)
  client_id         OAuth client that obtained the token (M2M apps)
  organization_id   Populated for org-scoped tokens
  scopes            List of granted permission strings
  audience          List of aud claim values
  github_token      GitHub access token extracted from the custom JWT claim
                    defined by LOGTO_GITHUB_TOKEN_CLAIM (default:
                    "github_access_token").  Falls back to the env-var
                    GITHUB_TOKEN when the claim is absent so the service
                    keeps working for M2M callers and during migration.

── Logto setup checklist ─────────────────────────────────────────────────────
1. Create an API Resource in the Logto console (e.g. https://api.yourapp.com)
   and define scopes (e.g. repo:summarize).

2. Add a GitHub social connector to your Logto tenant so users sign in with
   GitHub. Logto stores their GitHub access token internally.

3. Add a Custom JWT Claims script in Logto (Applications -> JWT claims):

     const getCustomJwtClaims = async ({ token, context }) => {
       return {
         github_access_token: context.social?.github?.accessToken ?? null,
       };
     };

4. Set environment variables:
     LOGTO_ENDPOINT=https://my-tenant.logto.app
     LOGTO_API_RESOURCE=https://api.yourapp.com
     LOGTO_REQUIRED_SCOPES=repo:summarize   (space-separated)
     LOGTO_GITHUB_TOKEN_CLAIM=github_access_token

5. In your client app, request the resource when obtaining tokens:
     resource: "https://api.yourapp.com"
     scope: "openid profile repo:summarize"
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from functools import lru_cache

import jwt
from fastapi import HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
_bearer_scheme = HTTPBearer(auto_error=False)


# ── AuthInfo ──────────────────────────────────────────────────────────────────


@dataclass
class AuthInfo:
    """Validated identity extracted from a Logto JWT (or static-key session)."""

    sub: str
    client_id: str | None = None
    organization_id: str | None = None
    scopes: list[str] = field(default_factory=list)
    audience: list[str] = field(default_factory=list)
    github_token: str | None = None  # per-user GitHub access token

    def requires_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def to_dict(self) -> dict:
        return {
            "sub": self.sub,
            "client_id": self.client_id,
            "organization_id": self.organization_id,
            "scopes": self.scopes,
            "audience": self.audience,
            # github_token deliberately excluded — never expose in responses
        }


# ── JWKS client (singleton per settings, handles key rotation) ───────────────


@lru_cache(maxsize=1)
def _get_jwks_client() -> PyJWKClient:
    settings = get_settings()
    jwks_uri = f"{settings.logto_endpoint.rstrip('/')}/oidc/jwks"
    logger.info("logto_jwks_client_init", jwks_uri=jwks_uri)
    return PyJWKClient(jwks_uri, cache_keys=True)


# ── JWT validation ────────────────────────────────────────────────────────────


def _validate_logto_jwt(token: str) -> AuthInfo:
    """
    Validate a Logto-issued JWT and return an AuthInfo.

    Verifies: signature (RS256/JWKS), issuer, audience, expiration, scopes.
    """
    settings = get_settings()
    issuer = f"{settings.logto_endpoint.rstrip('/')}/oidc"

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        payload: dict = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=issuer,
            options={"verify_aud": False},  # manual audience check below
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.error("logto_jwt_validation_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token validation failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Audience ──────────────────────────────────────────────────────────
    raw_aud = payload.get("aud", [])
    audience: list[str] = [raw_aud] if isinstance(raw_aud, str) else list(raw_aud)

    if settings.logto_api_resource and settings.logto_api_resource not in audience:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Token audience does not include the required API resource "
                f"({settings.logto_api_resource})."
            ),
        )

    # ── Scopes ────────────────────────────────────────────────────────────
    raw_scope: str = payload.get("scope", "") or ""
    granted_scopes = [s for s in raw_scope.split(" ") if s]

    if settings.logto_required_scopes:
        missing = [s for s in settings.logto_required_scopes if s not in granted_scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Token is missing required scopes: {missing}",
            )

    # ── GitHub token: custom claim -> env-var fallback ────────────────────
    github_token: str | None = (
        payload.get(settings.logto_github_token_claim) or settings.github_token or None
    )

    if not github_token:
        logger.warning(
            "github_token_missing",
            sub=payload.get("sub"),
            claim=settings.logto_github_token_claim,
            msg="No GitHub token found in JWT claim or GITHUB_TOKEN env var.",
        )

    return AuthInfo(
        sub=payload.get("sub", ""),
        client_id=payload.get("client_id"),
        organization_id=payload.get("organization_id"),
        scopes=granted_scopes,
        audience=audience,
        github_token=github_token,
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────


async def get_auth_info(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> AuthInfo:
    """
    FastAPI dependency. Returns a validated AuthInfo for each request.

    Auth mode is selected automatically based on which env vars are set.
    When both LOGTO_ENDPOINT and API_SECRET_KEY are configured, the static
    key is checked first (cheap comparison) before attempting Logto JWT
    validation.  This allows production Logto clients and simple API-key
    scripts to coexist.

      1. API_SECRET_KEY match  -> static-key AuthInfo (fast path)
      2. LOGTO_ENDPOINT set    -> Logto JWT validation
      3. Neither set           -> open access (local dev)
    """
    settings = get_settings()
    token = credentials.credentials if credentials else None

    # ── Static key fast path (works even when Logto is also configured) ───
    if settings.api_secret_key and token and secrets.compare_digest(
        token.encode(),
        settings.api_secret_key.encode(),
    ):
        return AuthInfo(sub="static-key-user", github_token=settings.github_token)

    # ── Logto JWT mode ────────────────────────────────────────────────────
    if settings.logto_endpoint:
        if token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header. Use: Bearer <logto_access_token>",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return _validate_logto_jwt(token)

    # ── Static key only (no Logto) ────────────────────────────────────────
    if settings.api_secret_key:
        if token is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header. Use: Bearer <api_secret_key>",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # If we reach here, the token didn't match the static key above
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    # ── Open / dev mode ───────────────────────────────────────────────────
    logger.warning(
        "auth_disabled",
        msg="No LOGTO_ENDPOINT or API_SECRET_KEY configured. All requests are unauthenticated.",
    )
    return AuthInfo(sub="anonymous", github_token=settings.github_token)


# Backward-compatible alias — the endpoint file depends on this name
verify_api_key = get_auth_info
