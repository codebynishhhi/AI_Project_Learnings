from pydantic import BaseModel, ConfigDict

class SummaryResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    summary : str
    key_points : list[str]


class SentimentResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')

    sentiment : str
    confidence_score : float
    explanation : str

class KeyPointsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_points : list[str]


class ActionItemsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_items : list[str]