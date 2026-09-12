from adaptive_rag.config import Settings


def test_settings_have_no_hardcoded_localhost():
    s = Settings(_env_file=None)
    for value in (s.opensearch_url, s.neo4j_uri, s.redis_url):
        assert "localhost" not in value
        assert "127.0.0.1" not in value


def test_settings_load_from_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://example-remote:6379")
    s = Settings(_env_file=None)
    assert s.redis_url == "redis://example-remote:6379"
