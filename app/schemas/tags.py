import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models import TagAssignedBy
from app.schemas.base import RequestSchema

# Upper bound on one request. Auto-tagging produces a handful; a person
# applies one or two. Twenty is generous and stops a malformed client from
# posting a thousand names in a loop.
MAX_TAGS_PER_REQUEST = 20


class TagRead(BaseModel):
    """A tag as the extension renders it."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    color: str


class TagWithCount(TagRead):
    """A tag plus how many documents use it, for the sidebar."""

    document_count: int


class DocumentTagRead(BaseModel):
    """
    One tag on one document, with its provenance.
    """

    tag: TagRead
    assigned_by: TagAssignedBy
    confidence: float | None


class AttachTagsRequestAttachTagsRequest(RequestSchema):
    names: list[str] = Field(min_length=1, max_length=MAX_TAGS_PER_REQUEST)
    model_config = ConfigDict(extra="forbid")
