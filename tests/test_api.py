"""Tests for API module."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


@pytest.fixture
def client():
    """Test client for API."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_ready(self, client):
        """Test readiness endpoint."""
        response = client.get("/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data


class TestQueryEndpoint:
    """Tests for query endpoint."""
    
    def test_query_basic(self, client):
        """Test basic query."""
        response = client.post("/query", json={
            "text": "What is Python?",
            "top_k": 5,
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "query_id" in data
        assert "answer" in data
    
    def test_query_with_threshold(self, client):
        """Test query with similarity threshold."""
        response = client.post("/query", json={
            "text": "What is Python?",
            "similarity_threshold": 0.5,
            "top_k": 10,
        })
        
        assert response.status_code == 200


class TestIngestEndpoint:
    """Tests for ingestion endpoint."""
    
    def test_ingest_records(self, client):
        """Test ingesting pre-embedded records."""
        response = client.post("/ingest", json={
            "records": [
                {
                    "metadata": {"name": "Test", "type": "document"},
                    "embedding": [0.1] * 768,
                },
            ],
            "source": "test",
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["ingested"] == 1
        assert len(data["chunk_ids"]) == 1


class TestAdminEndpoints:
    """Tests for admin endpoints."""
    
    def test_get_peers(self, client):
        """Test getting peers."""
        response = client.get("/admin/peers")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_hierarchy(self, client):
        """Test getting hierarchy info."""
        response = client.get("/admin/hierarchy")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_levels" in data
    
    def test_get_stats(self, client):
        """Test getting stats."""
        response = client.get("/admin/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "chunks" in data
