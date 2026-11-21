"""Embedding models and abstract base classes."""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import ClassVar


class BaseEmbeddingModel(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimension of the embedding model."""


class HashEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, lightweight n-dimensional feature-hashing embedding model.

    Computes normalized dense vectors from character/word n-grams and cross-lingual domain concepts.
    Suitable for zero-dependency local testing, fast unit tests, and multilingual MVP search.
    """

    # Domain-specific cross-lingual concept dictionary bridging EN, RU, and TG
    _CONCEPT_MAP: ClassVar[dict[str, list[str]]] = {
        # Charter / Bylaw
        "charter": ["charter", "устав", "оиннома"],
        "устав": ["charter", "устав", "оиннома"],
        "оиннома": ["charter", "устав", "оиннома"],
        "оинномаи": ["charter", "устав", "оиннома"],
        # Problem / Solution / Chaos
        "problem": ["problem", "problems", "проблема", "мушкилот", "хаос"],
        "problems": ["problem", "problems", "проблема", "мушкилот", "хаос"],
        "solve": ["solve", "solving", "solution", "решение", "ҳал"],
        "solving": ["solve", "solving", "solution", "решение", "ҳал"],
        "solution": ["solve", "solving", "solution", "решение", "ҳал"],
        "chaos": ["chaos", "хаос", "бетартибӣ"],
        "проблема": ["problem", "problems", "проблема", "мушкилот", "хаос"],
        "проблемы": ["problem", "problems", "проблема", "мушкилот", "хаос"],
        "решение": ["solve", "solving", "solution", "решение", "ҳал"],
        "хаос": ["chaos", "problem", "проблема", "хаос"],
        "мушкилот": ["problem", "problems", "проблема", "мушкилот"],
        "мушкилоти": ["problem", "problems", "проблема", "мушкилот"],
        # Hadaf branding & Mission
        "hadaf": ["hadaf", "ҳадаф", "хадаф", "мақсад", "цель"],
        "ҳадаф": ["hadaf", "ҳадаф", "хадаф", "мақсад", "цель"],
        "хадаф": ["hadaf", "ҳадаф", "хадаф", "мақсад", "цель"],
        "mission": ["mission", "мақсад", "миссия"],
        "миссия": ["mission", "мақсад", "миссия"],
        # Commission / Fees
        "commission": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "fee": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "fees": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "комиссия": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "комиссияи": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "процент": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        "фоиз": ["commission", "комиссия", "фоиз", "fee", "fees", "процент"],
        # Foundations & NGOs
        "foundation": ["foundation", "foundations", "fund", "funds", "фонд", "нко"],
        "foundations": ["foundation", "foundations", "fund", "funds", "фонд", "нко"],
        "fund": ["foundation", "foundations", "fund", "funds", "фонд", "нко"],
        "funds": ["foundation", "foundations", "fund", "funds", "фонд", "нко"],
        "фонд": ["foundation", "fund", "нко", "фонд"],
        "фонды": ["foundation", "fund", "нко", "фонд"],
        # Verification & Moderation
        "verification": [
            "verification",
            "verified",
            "verify",
            "верификация",
            "проверка",
            "тасдиқ",
        ],
        "verified": [
            "verification",
            "verified",
            "verify",
            "верифицирован",
            "проверен",
            "тасдиқшуда",
        ],
        "verify": [
            "verification",
            "verified",
            "verify",
            "проверить",
            "верифицировать",
            "тасдиқ",
        ],
        "верификация": ["verification", "проверка", "тасдиқ", "верификация"],
        "проверка": ["verification", "проверка", "тасдиқ"],
        # Donations & Transparency
        "donation": [
            "donation",
            "donations",
            "charity",
            "пожертвование",
            "хайрия",
            "садақа",
            "садака",
        ],
        "donations": [
            "donation",
            "donations",
            "charity",
            "пожертвование",
            "пожертвования",
            "хайрия",
            "садақа",
            "садака",
        ],
        "charity": ["donation", "charity", "благотворительность", "хайрия"],
        "пожертвование": [
            "donation",
            "donations",
            "charity",
            "пожертвование",
            "хайрия",
            "садақа",
        ],
        "пожертвования": [
            "donation",
            "donations",
            "charity",
            "пожертвование",
            "пожертвования",
            "хайрия",
            "садақа",
        ],
        "хайрия": ["donation", "charity", "пожертвование", "хайрия"],
        "receipt": ["receipt", "receipts", "чек", "чеки", "квитанция"],
        "receipts": ["receipt", "receipts", "чек", "чеки", "квитанция"],
        "чек": ["receipt", "receipts", "квитанция", "чек"],
        "чеки": ["receipt", "receipts", "квитанция", "чеки"],
        "transparency": ["transparency", "прозрачность", "шаффофият"],
        "прозрачность": ["transparency", "шаффофият", "прозрачность"],
        # Institutions & Campaigns
        "institution": ["institution", "учреждение", "детдом", "интернат", "муассиса"],
        "orphanage": ["orphanage", "детдом", "интернат", "муассиса"],
        "учреждение": ["institution", "детдом", "интернат", "муассиса", "учреждение"],
        "campaign": ["campaign", "campaigns", "сбор", "кампания", "ҷамъоварӣ"],
        "campaigns": ["campaign", "campaigns", "сборы", "кампании", "ҷамъоварӣ"],
        "сбор": ["campaign", "campaigns", "ҷамъоварӣ", "сбор"],
        "сборы": ["campaign", "campaigns", "ҷамъоварӣ", "сборы"],
        # Stats & Events
        "statistics": ["statistics", "stats", "статистика", "омор"],
        "stats": ["statistics", "stats", "статистика", "омор"],
        "статистика": ["statistics", "stats", "омор", "статистика"],
        "омор": ["statistics", "stats", "статистика", "омор"],
        "event": ["event", "events", "мероприятие", "событие", "чорабинӣ"],
        "events": ["event", "events", "мероприятия", "события", "чорабиниҳо"],
    }

    def __init__(self, dimension: int = 128) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase words, cross-lingual concepts, and sub-word n-grams."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = [w for w in cleaned.split() if w]
        tokens: list[str] = list(words)

        # Cross-lingual concept expansion
        for w in words:
            if w in self._CONCEPT_MAP:
                tokens.extend(self._CONCEPT_MAP[w])

        # Add character 3-grams for subword matching (crucial for Russian/Tajik/Uzbek morphology)
        for w in list(tokens):
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append(w[i : i + 3])
        return tokens

    def _embed_single(self, text: str) -> list[float]:
        """Compute normalized dense hash vector."""
        vector = [0.0] * self._dim
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            # Deterministic MD5 hash to bucket
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vector[idx] += sign

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_single(t) for t in texts]
