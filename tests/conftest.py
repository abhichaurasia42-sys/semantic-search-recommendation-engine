import pytest

from semantic_search.corpus import Document


@pytest.fixture
def tiny_corpus() -> list[Document]:
    return [
        Document(
            id="doc-1",
            domain="technology",
            subtopic="cloud",
            title="Cloud Basics",
            text="Cloud computing provides on-demand compute and storage over the internet.",
        ),
        Document(
            id="doc-2",
            domain="technology",
            subtopic="cloud",
            title="Serverless Overview",
            text="Serverless functions run code without provisioning dedicated servers, billed by execution time.",
        ),
        Document(
            id="doc-3",
            domain="technology",
            subtopic="ml",
            title="Neural Networks",
            text="Neural networks learn patterns from data using layers of weighted connections.",
        ),
        Document(
            id="doc-4",
            domain="cooking",
            subtopic="baking",
            title="Bread Baking",
            text="Baking bread requires yeast fermentation and careful control of oven temperature.",
        ),
        Document(
            id="doc-5",
            domain="cooking",
            subtopic="baking",
            title="Pastry Dough",
            text="Cold butter in pastry dough creates flaky layers when baked in a hot oven.",
        ),
        Document(
            id="doc-6",
            domain="cooking",
            subtopic="knives",
            title="Knife Sharpening",
            text="A sharp chef's knife makes chopping vegetables safer and more precise.",
        ),
    ]
