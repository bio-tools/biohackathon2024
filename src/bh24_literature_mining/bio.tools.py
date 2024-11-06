
from dataclasses import dataclass
from typing import List


@dataclass
class Biotool:

    biotool_id: str
    name: str
    operations: List[str] = []
    topics: List[str] = []