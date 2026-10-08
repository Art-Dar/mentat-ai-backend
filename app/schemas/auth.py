from pydantic import BaseModel
from app.schemas.base import RequestSchemal


class LoginRequest(RequestSchema):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
