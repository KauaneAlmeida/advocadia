"""
Request Models

Pydantic models for validating incoming requests to the API.
These models ensure data integrity and provide automatic documentation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional

class ChatRequest(BaseModel):
    """
    Model for chat message requests.
    
    Attributes:
        message (str): The user's chat message
        user_id (Optional[str]): Optional user identifier for tracking
        session_id (Optional[str]): Optional session identifier
    """
    
    message: str = Field(
        ...,
        description="The chat message from the user",
        min_length=1,
        max_length=4000,
        example="Hello, how can you help me today?"
    )
    
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier",
        max_length=100,
        example="user_12345"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Optional session identifier for conversation tracking",
        max_length=100,
        example="session_abc123"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """Validate that the message is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()
    
    @validator('user_id', 'session_id')
    def validate_optional_ids(cls, v):
        """Validate optional ID fields."""
        if v is not None:
            return v.strip() if v.strip() else None
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "message": "Hello, I need help with my project",
                "user_id": "user_12345",
                "session_id": "session_abc123"
            }
        }

class ConversationRequest(BaseModel):
    """
    Model for conversation flow requests.
    
    Attributes:
        message (str): The user's response to current question
        session_id (Optional[str]): Session identifier for conversation tracking
    """
    
    message: str = Field(
        ...,
        description="The user's response to the current question",
        min_length=1,
        max_length=2000,
        example="John Smith"
    )
    
    session_id: Optional[str] = Field(
        None,
        description="Session identifier for conversation tracking",
        max_length=100,
        example="session_abc123"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """Validate that the message is not just whitespace."""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty or just whitespace')
        return v.strip()

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "message": "John Smith",
                "session_id": "session_abc123"
            }
        }