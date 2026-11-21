"""Unit tests for deterministic cross-lingual HashEmbeddings and InMemoryVectorStore."""

import math

import pytest

from app.rag.embeddings import HashEmbeddingModel
from app.rag.models import Chunk
from app.rag.vector_store import InMemoryVectorStore


@pytest.mark.asyncio
async def test_hash_embeddings_deterministic_and_dimensions():
    """Test deterministic hash embedding generation and dimensions."""
    # Default 128 dimensions
    model = HashEmbeddingModel()
    assert model.dimension == 128

    vec1 = await model.embed_query("charity transparency in Tajikistan")
    vec2 = await model.embed_query("charity transparency in Tajikistan")
    vec3 = await model.embed_query("completely different topic")

    assert len(vec1) == 128
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Distinct inputs produce different vectors

    # Custom dimension
    model64 = HashEmbeddingModel(dimension=64)
    assert model64.dimension == 64
    vec64 = await model64.embed_query("charity transparency in Tajikistan")
    assert len(vec64) == 64

    # Empty text produces zero vector
    vec_empty = await model.embed_query("")
    assert vec_empty == [0.0] * 128
    vec_whitespace = await model.embed_query("   \t\n  ")
    assert vec_whitespace == [0.0] * 128


@pytest.mark.asyncio
async def test_hash_embeddings_normalization():
    """Test that embedding vectors are L2-normalized."""
    model = HashEmbeddingModel(dimension=128)
    samples = [
        "charter",
        "устав",
        "оиннома",
        "problem solving in community",
        "мушкилот ва ҳалли он",
        "Hadaf open-source charitable foundation",
    ]

    for text in samples:
        vec = await model.embed_query(text)
        norm = math.sqrt(sum(v * v for v in vec))
        assert math.isclose(norm, 1.0, rel_tol=1e-5), (
            f"Vector for '{text}' is not L2-normalized (norm={norm})"
        )


@pytest.mark.asyncio
async def test_hash_embeddings_embed_documents():
    """Test batch document embedding produces consistent vectors."""
    model = HashEmbeddingModel(dimension=128)
    docs = [
        "Hadaf mission and charter",
        "Zero commission charity donations",
        "Transparent report receipts and verification",
    ]

    batch_vectors = await model.embed_documents(docs)
    assert len(batch_vectors) == len(docs)

    for doc_text, batch_vec in zip(docs, batch_vectors):
        single_vec = await model.embed_query(doc_text)
        assert batch_vec == single_vec


@pytest.mark.asyncio
async def test_cross_lingual_concept_mappings():
    """Test exact required concept mappings across EN, RU, and TG:
    - charter ↔ устав ↔ оиннома
    - problem ↔ проблема ↔ мушкилот
    - commission ↔ комиссия ↔ фоиз
    """
    model = HashEmbeddingModel(dimension=128)
    unrelated_text = "weather forecast sunshine precipitation temperature"
    unrelated_vec = await model.embed_query(unrelated_text)

    concept_triads = [
        ("charter", "устав", "оиннома"),
        ("problem", "проблема", "мушкилот"),
        ("commission", "комиссия", "фоиз"),
    ]

    def dot_product(v1: list[float], v2: list[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    for en, ru, tg in concept_triads:
        vec_en = await model.embed_query(en)
        vec_ru = await model.embed_query(ru)
        vec_tg = await model.embed_query(tg)

        sim_en_ru = dot_product(vec_en, vec_ru)
        sim_en_tg = dot_product(vec_en, vec_tg)
        sim_ru_tg = dot_product(vec_ru, vec_tg)

        # Cross-lingual synonyms must show high similarity (> 0.70)
        assert sim_en_ru > 0.70, f"Low similarity between {en} and {ru}: {sim_en_ru}"
        assert sim_en_tg > 0.70, f"Low similarity between {en} and {tg}: {sim_en_tg}"
        assert sim_ru_tg > 0.70, f"Low similarity between {ru} and {tg}: {sim_ru_tg}"

        # Cross-lingual synonyms must be significantly more similar than unrelated text
        sim_unrelated = dot_product(vec_en, unrelated_vec)
        assert sim_en_ru > sim_unrelated + 0.4
        assert sim_en_tg > sim_unrelated + 0.4


@pytest.mark.asyncio
async def test_in_memory_vector_store_crud_and_validation():
    """Test CRUD operations, chunk retrieval, count, clear, and validation."""
    model = HashEmbeddingModel(dimension=64)
    store = InMemoryVectorStore()

    assert store.count() == 0

    chunk1 = Chunk(
        id="chunk-1",
        content="First test chunk",
        metadata={"category": "test"},
        embedding=await model.embed_query("First test chunk"),
    )
    chunk2 = Chunk(
        id="chunk-2",
        content="Second test chunk",
        metadata={"category": "test"},
        embedding=await model.embed_query("Second test chunk"),
    )

    # Adding valid chunks
    added = await store.add_chunks([chunk1, chunk2])
    assert added == ["chunk-1", "chunk-2"]
    assert store.count() == 2

    # Retrieval by ID
    retrieved = await store.get_chunk("chunk-1")
    assert retrieved is not None
    assert retrieved.id == "chunk-1"
    assert retrieved.content == "First test chunk"

    # Nonexistent chunk returns None
    assert await store.get_chunk("chunk-nonexistent") is None

    # Adding chunk without embedding raises ValueError
    chunk_no_embed = Chunk(id="chunk-no-embed", content="No vector")
    with pytest.raises(ValueError, match="missing an embedding vector"):
        await store.add_chunks([chunk_no_embed])

    # Clear
    await store.clear()
    assert store.count() == 0
    assert await store.get_chunk("chunk-1") is None


@pytest.mark.asyncio
async def test_in_memory_vector_store_similarity_ranking():
    """Test that similarity search ranks chunks in descending score order and respects top_k."""
    model = HashEmbeddingModel(dimension=128)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(
            id="c_charity",
            content="Open source charity platform in Tajikistan with full transparency",
            metadata={"domain": "charity"},
            embedding=await model.embed_query(
                "Open source charity platform in Tajikistan with full transparency"
            ),
        ),
        Chunk(
            id="c_finance",
            content="Zero transaction commission fee policy and financial reporting",
            metadata={"domain": "finance"},
            embedding=await model.embed_query(
                "Zero transaction commission fee policy and financial reporting"
            ),
        ),
        Chunk(
            id="c_orphanage",
            content="Volunteers visiting orphanages and boarding schools",
            metadata={"domain": "social"},
            embedding=await model.embed_query(
                "Volunteers visiting orphanages and boarding schools"
            ),
        ),
    ]
    await store.add_chunks(chunks)

    # Query finance topic
    query = "What is the transaction commission fee?"
    query_vec = await model.embed_query(query)

    # Top 3 results
    results = await store.similarity_search(query_vec, top_k=3)
    assert len(results) == 3
    # Check descending order
    assert results[0].score >= results[1].score >= results[2].score
    # Most relevant chunk should be finance
    assert results[0].chunk.id == "c_finance"

    # Test top_k limiting
    top_1 = await store.similarity_search(query_vec, top_k=1)
    assert len(top_1) == 1
    assert top_1[0].chunk.id == "c_finance"

    top_0 = await store.similarity_search(query_vec, top_k=0)
    assert len(top_0) == 0


@pytest.mark.asyncio
async def test_in_memory_vector_store_min_score_filter():
    """Test min_score threshold filtering in similarity search."""
    model = HashEmbeddingModel(dimension=128)
    store = InMemoryVectorStore()

    chunk = Chunk(
        id="c1",
        content="Zero commission policy",
        embedding=await model.embed_query("Zero commission policy"),
    )
    await store.add_chunks([chunk])

    query_vec = await model.embed_query("commission fee")
    results = await store.similarity_search(query_vec, top_k=5, min_score=0.0)
    assert len(results) == 1
    actual_score = results[0].score

    # Filter with higher min_score than actual score
    strict_results = await store.similarity_search(
        query_vec, top_k=5, min_score=actual_score + 0.1
    )
    assert len(strict_results) == 0

    # Filter with lower min_score
    lenient_results = await store.similarity_search(
        query_vec, top_k=5, min_score=actual_score - 0.1
    )
    assert len(lenient_results) == 1


@pytest.mark.asyncio
async def test_in_memory_vector_store_metadata_filtering():
    """Test single and multi-attribute metadata filtering."""
    model = HashEmbeddingModel(dimension=64)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(
            id="doc1",
            content="Tajik language charter documentation",
            metadata={"lang": "tg", "type": "charter", "verified": True},
            embedding=await model.embed_query("Tajik language charter documentation"),
        ),
        Chunk(
            id="doc2",
            content="Russian language charter documentation",
            metadata={"lang": "ru", "type": "charter", "verified": True},
            embedding=await model.embed_query("Russian language charter documentation"),
        ),
        Chunk(
            id="doc3",
            content="English language charter draft unverified",
            metadata={"lang": "en", "type": "charter", "verified": False},
            embedding=await model.embed_query(
                "English language charter draft unverified"
            ),
        ),
    ]
    await store.add_chunks(chunks)

    query_vec = await model.embed_query("charter documentation")

    # Filter by single metadata key
    ru_results = await store.similarity_search(
        query_vec, top_k=5, filter_metadata={"lang": "ru"}
    )
    assert len(ru_results) == 1
    assert ru_results[0].chunk.id == "doc2"

    # Filter by multiple metadata keys
    verified_charters = await store.similarity_search(
        query_vec,
        top_k=5,
        filter_metadata={"type": "charter", "verified": True},
    )
    assert len(verified_charters) == 2
    assert {r.chunk.id for r in verified_charters} == {"doc1", "doc2"}

    # Filter with non-matching metadata
    no_results = await store.similarity_search(
        query_vec, top_k=5, filter_metadata={"lang": "uz"}
    )
    assert len(no_results) == 0


@pytest.mark.asyncio
async def test_cross_lingual_retrieval_end_to_end():
    """End-to-end test: English queries retrieve Russian and Tajik documents correctly."""
    model = HashEmbeddingModel(dimension=128)
    store = InMemoryVectorStore()

    chunks = [
        Chunk(
            id="ru_charter",
            content="Устав благотворительного фонда Хадаф определяет правила и структуру",
            metadata={"lang": "ru", "topic": "charter"},
            embedding=await model.embed_query(
                "Устав благотворительного фонда Хадаф определяет правила и структуру"
            ),
        ),
        Chunk(
            id="tg_problem",
            content="Мушкилоти асосии созмонҳо набудани шаффофият ва эътимоди мардум мебошад",
            metadata={"lang": "tg", "topic": "problem"},
            embedding=await model.embed_query(
                "Мушкилоти асосии созмонҳо набудани шаффофият ва эътимоди мардум мебошад"
            ),
        ),
        Chunk(
            id="tg_commission",
            content="Платформаи Ҳадаф сифр фоиз комиссия мегирад ва ҳамаи маблағ мерасад",
            metadata={"lang": "tg", "topic": "commission"},
            embedding=await model.embed_query(
                "Платформаи Ҳадаф сифр фоиз комиссия мегирад ва ҳамаи маблағ мерасад"
            ),
        ),
    ]
    await store.add_chunks(chunks)

    # 1. Search in English for charter
    query_charter = await model.embed_query("Where can I read the foundation charter?")
    res_charter = await store.similarity_search(query_charter, top_k=1)
    assert len(res_charter) == 1
    assert res_charter[0].chunk.id == "ru_charter"

    # 2. Search in English for problem
    query_problem = await model.embed_query("What is the main problem?")
    res_problem = await store.similarity_search(query_problem, top_k=1)
    assert len(res_problem) == 1
    assert res_problem[0].chunk.id == "tg_problem"

    # 3. Search in English for commission
    query_commission = await model.embed_query(
        "What is the platform commission fee percentage?"
    )
    res_commission = await store.similarity_search(query_commission, top_k=1)
    assert len(res_commission) == 1
    assert res_commission[0].chunk.id == "tg_commission"
