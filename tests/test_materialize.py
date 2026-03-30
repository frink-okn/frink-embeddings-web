from frink_embeddings_web.indexing.index import fallback_label, humanize


def test_humanize_text():
    assert humanize("hello_world") == "hello world"
    assert humanize("strip-till") == "strip till"
    assert humanize("ProjectScenario") == "Project Scenario"


def test_url_fallback():
    assert fallback_label("http://example.com/FirstName") == "First Name"
    assert fallback_label(
        "http://example.com/ontology#hasAttribute"
    ) == "has Attribute"
