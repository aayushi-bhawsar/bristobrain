"""
api/main.py
FastAPI application entrypoint for BistroBrain.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import date

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import action_cards, query, webhook

log = structlog.get_logger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")
RESTAURANT_NAME = os.getenv("RESTAURANT_NAME", "BistroBrain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("bistrobrain_starting", env=APP_ENV, restaurant=RESTAURANT_NAME)
    yield
    log.info("bistrobrain_stopping")


app = FastAPI(
    title="BistroBrain API",
    description=(
        "AI Store Manager for independent restaurants. "
        "Turns POS data into daily action cards and natural language insights."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(action_cards.router, prefix="/api", tags=["Action Cards"])
app.include_router(query.router, prefix="/api", tags=["Q&A"])
app.include_router(webhook.router, prefix="/api", tags=["WhatsApp Webhook"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "BistroBrain",
        "restaurant": RESTAURANT_NAME,
        "status": "online",
        "date": str(date.today()),
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
