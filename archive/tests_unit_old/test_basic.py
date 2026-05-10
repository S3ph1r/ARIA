"""
Basic tests for ARIA server
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gpu_available" in data
    assert "gpu_count" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ARIA GPU Inference Broker"
    assert data["status"] == "running"

def test_tts_voices_placeholder():
    """Test TTS voices endpoint (placeholder)"""
    response = client.get("/tts/voices")
    assert response.status_code == 200
    data = response.json()
    assert "voices" in data
    assert data["message"] == "TTS backend not initialized"

def test_tts_synthesize_placeholder():
    """Test TTS synthesis endpoint (placeholder)"""
    response = client.post("/tts/synthesize", params={"text": "test"})
    assert response.status_code == 503  # Service unavailable
    data = response.json()
    assert "TTS backend not available" in data["detail"]