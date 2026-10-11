from pydantic import BaseModel, Field

from app.schemas.base import RequestSchema


class LoginRequest(RequestSchema):
    email: str
    password: str


class RegisterRequest(RequestSchema):
    email: str
    password: str = Field(min_length=8)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
