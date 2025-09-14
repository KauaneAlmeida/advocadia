"""
Response Models

Pydantic models for API responses. These models ensure consistent
response formats and provide automatic documentation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatResponse(BaseModel):
    """
    Model for chat message responses.
    
    Attributes:
        reply (str): The AI-generated response to the user's message
        timestamp (Optional[datetime]): When the response was generated
        model_used (Optional[str]): Which AI model generated the response
        confidence (Optional[float]): Confidence score of the response
    """
    
    reply: str = Field(
        ...,
        description="The AI-generated response",
        example="Com base na sua situação, recomendo que você procure um advogado especializado em direito trabalhista..."
    )
    
    timestamp: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="When the response was generated"
    )
    
    model_used: Optional[str] = Field(
        default="gemini-legal",
        description="The AI model that generated this response",
        example="gemini-1.5-flash"
    )
    
    confidence: Optional[float] = Field(
        default=1.0,
        description="Confidence score for the response (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        example=0.95
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "reply": "Com base na sua situação, recomendo que você procure um advogado especializado em direito trabalhista para analisar seu caso detalhadamente.",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "model_used": "gemini-1.5-flash",
                "confidence": 0.95
            }
        }

class ErrorResponse(BaseModel):
    """
    Model for error responses.
    
    Attributes:
        error (bool): Always True for error responses
        message (str): Human-readable error message
        status_code (int): HTTP status code
        details (Optional[str]): Additional error details
    """
    
    error: bool = Field(
        default=True,
        description="Indicates this is an error response"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Validation error"
    )
    
    status_code: int = Field(
        ...,
        description="HTTP status code",
        example=400
    )
    
    details: Optional[str] = Field(
        None,
        description="Additional error details",
        example="Message field is required"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "error": True,
                "message": "Validation error",
                "status_code": 400,
                "details": "Message field is required"
            }
        }

class HealthResponse(BaseModel):
    """
    Model for health check responses.
    
    Attributes:
        status (str): Health status of the service
        message (str): Descriptive message about the service status
        timestamp (datetime): When the health check was performed
    """
    
    status: str = Field(
        ...,
        description="Health status",
        example="healthy"
    )
    
    message: str = Field(
        ...,
        description="Status description",
        example="Law Firm AI Chat Backend is running"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the health check was performed"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "message": "Law Firm AI Chat Backend is running",
                "timestamp": "2024-01-15T10:30:00.000Z"
            }
        }

class ConversationResponse(BaseModel):
    """
    Model for conversation flow responses.
    
    Attributes:
        session_id (str): Session identifier
        question (Optional[str]): Next question in the flow
        response (Optional[str]): AI response (when in AI mode)
        step_id (Optional[int]): Current step ID
        is_final_step (Optional[bool]): Whether this is the final step
        flow_completed (bool): Whether the guided flow is completed
        ai_mode (bool): Whether conversation is in AI mode
        phone_collected (bool): Whether phone number was collected
        collecting_phone (Optional[bool]): Whether currently collecting phone
        redirect_message (Optional[bool]): Whether this is a redirect for irrelevant response
        validation_error (Optional[bool]): Whether there was a validation error
        lead_saved (Optional[bool]): Whether lead data was saved
        lead_id (Optional[str]): ID of saved lead
        whatsapp_sent (Optional[bool]): Whether WhatsApp message was sent
        phone_number (Optional[str]): Collected phone number
    """
    
    session_id: str = Field(
        ...,
        description="Session identifier",
        example="session_abc123"
    )
    
    question: Optional[str] = Field(
        None,
        description="Next question in the guided flow",
        example="Qual é o seu nome completo?"
    )
    
    response: Optional[str] = Field(
        None,
        description="AI response when in AI mode",
        example="Com base na sua situação, recomendo que você procure um advogado especializado..."
    )
    
    step_id: Optional[int] = Field(
        None,
        description="Current step ID in the flow",
        example=1
    )
    
    is_final_step: Optional[bool] = Field(
        False,
        description="Whether this is the final step in the flow"
    )
    
    flow_completed: bool = Field(
        False,
        description="Whether the guided flow is completed"
    )
    
    ai_mode: bool = Field(
        False,
        description="Whether conversation is in AI mode"
    )
    
    phone_collected: bool = Field(
        False,
        description="Whether phone number was collected"
    )
    
    collecting_phone: Optional[bool] = Field(
        None,
        description="Whether currently collecting phone number"
    )
    
    redirect_message: Optional[bool] = Field(
        None,
        description="Whether this is a redirect for irrelevant response"
    )
    
    validation_error: Optional[bool] = Field(
        None,
        description="Whether there was a validation error"
    )
    
    lead_saved: Optional[bool] = Field(
        None,
        description="Whether lead data was successfully saved"
    )
    
    lead_id: Optional[str] = Field(
        None,
        description="ID of the saved lead record",
        example="lead_xyz789"
    )
    
    whatsapp_sent: Optional[bool] = Field(
        None,
        description="Whether WhatsApp message was sent successfully"
    )
    
    phone_number: Optional[str] = Field(
        None,
        description="Collected phone number",
        example="11999999999"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "session_id": "session_abc123",
                "question": "Qual é o seu nome completo?",
                "step_id": 1,
                "is_final_step": False,
                "flow_completed": False,
                "ai_mode": False,
                "phone_collected": False
            }
        }