from pydantic import BaseModel, Field, model_validator

from app.schemas.base import RequestSchema


class LoginRequest(RequestSchema):
    email: str
    password: str


class RegisterRequest(RequestSchema):
    email: str
    password: str = Field(min_length=8)


class GoogleAuthRequest(RequestSchema):
    # id_token: a Google Identity Services / OAuth Playground credential.
    # access_token: what chrome.identity.getAuthToken returns — the Chrome
    # extension's actual flow. Exactly one, since they're verified
    # differently (a local JWT check vs. a call to Google's tokeninfo).
    id_token: str | None = None
    access_token: str | None = None

    @model_validator(mode="after")
    def _exactly_one_token(self) -> "GoogleAuthRequest":
        if bool(self.id_token) == bool(self.access_token):
            raise ValueError("Provide exactly one of id_token or access_token")
        return self


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ConnectResponse(BaseModel):
    provider: str
    connected: bool = True
