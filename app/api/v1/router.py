"""
app/api/v1/router.py
─────────────────────
Aggregate all v1 endpoint routers under a single prefix.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.summarize import router as summarize_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(summarize_router)
