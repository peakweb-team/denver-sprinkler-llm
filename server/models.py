"""Pydantic request / response schemas for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation turn."""

    role: str = Field(
        ...,
        description="Role of the message author: 'user' or 'assistant'.",
        pattern=r"^(user|assistant)$",
    )
    content: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text content of the message.",
    )


class ChatRequest(BaseModel):
    """Incoming chat request with conversation history."""

    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Ordered list of conversation messages.",
    )


class ChatResponse(BaseModel):
    """Server response containing the assistant reply."""

    response: str


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str
    model: str
    version: str
