from typing import List

from pydantic import BaseModel


class LogContent(BaseModel):
    name: str
    lines: List[str]
