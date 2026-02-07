"""YAML config loader with environment variable overlay.

Env vars take precedence over YAML values.
Env var naming: DACHRAG__{section}__{key} (double underscore separator)
e.g., DACHRAG__DATABASE__HOST overrides database.host
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "dachrag"
    user: str = "dachrag"
    password: str = "password"
    pool_size: int = 10

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class EmbeddingsConfig:
    provider: str = "sentence-transformers"
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    dimension: int = 768
    batch_size: int = 64


@dataclass
class LLMConfig:
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class ClusteringConfig:
    leiden_resolution: list[float] = field(default_factory=lambda: [2.0, 1.0, 0.5])
    min_community_size: int = 3
    knn_k: int = 10
    knn_threshold: float = 0.5
    silhouette_threshold: float = 0.3
    max_hierarchy_levels: int = 5


@dataclass
class IndexingConfig:
    ef_search: int = 64
    ef_construction: int = 128
    num_neighbors: int = 32


@dataclass
class RoutingConfig:
    similarity_threshold: float = 0.35
    min_communities_per_layer: int = 1
    max_communities_per_layer: int = 20
    top_k_results: int = 10


@dataclass
class NetworkingConfig:
    grpc_port: int = 50051
    max_peers: int = 100
    heartbeat_interval: int = 30
    coordinator_address: str = "localhost:50050"


@dataclass
class MaintenanceConfig:
    centroid_update_threshold: int = 1000
    canary_interval_minutes: int = 30
    recluster_cooldown_seconds: int = 3600


@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    networking: NetworkingConfig = field(default_factory=NetworkingConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)


_settings: Settings | None = None


def get_settings(config_path: str = "config/default.yaml") -> Settings:
    """Get the global settings instance, loading from config if not yet loaded."""
    global _settings
    if _settings is None:
        _settings = _load_settings(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the cached settings (useful for testing)."""
    global _settings
    _settings = None


def _load_settings(config_path: str) -> Settings:
    """Load settings from YAML file with env var overlay."""
    settings = Settings()
    
    # Load YAML if exists
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file) as f:
            yaml_config = yaml.safe_load(f) or {}
        settings = _apply_yaml(settings, yaml_config)
    
    # Overlay environment variables
    settings = _apply_env_vars(settings)
    
    return settings


def _apply_yaml(settings: Settings, yaml_config: dict[str, Any]) -> Settings:
    """Apply YAML config values to settings."""
    if "database" in yaml_config:
        db = yaml_config["database"]
        settings.database = DatabaseConfig(
            host=db.get("host", settings.database.host),
            port=db.get("port", settings.database.port),
            name=db.get("name", settings.database.name),
            user=db.get("user", settings.database.user),
            password=db.get("password", settings.database.password),
            pool_size=db.get("pool_size", settings.database.pool_size),
        )
    
    if "embeddings" in yaml_config:
        emb = yaml_config["embeddings"]
        settings.embeddings = EmbeddingsConfig(
            provider=emb.get("provider", settings.embeddings.provider),
            model=emb.get("model", settings.embeddings.model),
            dimension=emb.get("dimension", settings.embeddings.dimension),
            batch_size=emb.get("batch_size", settings.embeddings.batch_size),
        )
    
    if "llm" in yaml_config:
        llm = yaml_config["llm"]
        settings.llm = LLMConfig(
            provider=llm.get("provider", settings.llm.provider),
            model=llm.get("model", settings.llm.model),
            temperature=llm.get("temperature", settings.llm.temperature),
            max_tokens=llm.get("max_tokens", settings.llm.max_tokens),
        )
    
    if "clustering" in yaml_config:
        cl = yaml_config["clustering"]
        settings.clustering = ClusteringConfig(
            leiden_resolution=cl.get("leiden_resolution", settings.clustering.leiden_resolution),
            min_community_size=cl.get("min_community_size", settings.clustering.min_community_size),
            knn_k=cl.get("knn_k", settings.clustering.knn_k),
            knn_threshold=cl.get("knn_threshold", settings.clustering.knn_threshold),
            silhouette_threshold=cl.get("silhouette_threshold", settings.clustering.silhouette_threshold),
            max_hierarchy_levels=cl.get("max_hierarchy_levels", settings.clustering.max_hierarchy_levels),
        )
    
    if "indexing" in yaml_config:
        idx = yaml_config["indexing"]
        settings.indexing = IndexingConfig(
            ef_search=idx.get("ef_search", settings.indexing.ef_search),
            ef_construction=idx.get("ef_construction", settings.indexing.ef_construction),
            num_neighbors=idx.get("num_neighbors", settings.indexing.num_neighbors),
        )
    
    if "routing" in yaml_config:
        rt = yaml_config["routing"]
        settings.routing = RoutingConfig(
            similarity_threshold=rt.get("similarity_threshold", settings.routing.similarity_threshold),
            min_communities_per_layer=rt.get("min_communities_per_layer", settings.routing.min_communities_per_layer),
            max_communities_per_layer=rt.get("max_communities_per_layer", settings.routing.max_communities_per_layer),
            top_k_results=rt.get("top_k_results", settings.routing.top_k_results),
        )
    
    if "networking" in yaml_config:
        net = yaml_config["networking"]
        settings.networking = NetworkingConfig(
            grpc_port=net.get("grpc_port", settings.networking.grpc_port),
            max_peers=net.get("max_peers", settings.networking.max_peers),
            heartbeat_interval=net.get("heartbeat_interval", settings.networking.heartbeat_interval),
            coordinator_address=net.get("coordinator_address", settings.networking.coordinator_address),
        )
    
    if "maintenance" in yaml_config:
        mnt = yaml_config["maintenance"]
        settings.maintenance = MaintenanceConfig(
            centroid_update_threshold=mnt.get("centroid_update_threshold", settings.maintenance.centroid_update_threshold),
            canary_interval_minutes=mnt.get("canary_interval_minutes", settings.maintenance.canary_interval_minutes),
            recluster_cooldown_seconds=mnt.get("recluster_cooldown_seconds", settings.maintenance.recluster_cooldown_seconds),
        )
    
    return settings


def _apply_env_vars(settings: Settings) -> Settings:
    """Apply environment variable overrides. Format: DACHRAG__SECTION__KEY."""
    prefix = "DACHRAG__"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        parts = key[len(prefix):].lower().split("__")
        if len(parts) != 2:
            continue
        
        section, field_name = parts
        _set_field(settings, section, field_name, value)
    
    return settings


def _set_field(settings: Settings, section: str, field_name: str, value: str) -> None:
    """Set a field on the settings object from an env var value."""
    section_map = {
        "database": settings.database,
        "embeddings": settings.embeddings,
        "llm": settings.llm,
        "clustering": settings.clustering,
        "indexing": settings.indexing,
        "routing": settings.routing,
        "networking": settings.networking,
        "maintenance": settings.maintenance,
    }
    
    section_obj = section_map.get(section)
    if section_obj is None:
        return
    
    if not hasattr(section_obj, field_name):
        return
    
    current_value = getattr(section_obj, field_name)
    
    # Type coercion based on current type
    if isinstance(current_value, bool):
        setattr(section_obj, field_name, value.lower() in ("true", "1", "yes"))
    elif isinstance(current_value, int):
        setattr(section_obj, field_name, int(value))
    elif isinstance(current_value, float):
        setattr(section_obj, field_name, float(value))
    elif isinstance(current_value, list):
        # For lists, expect comma-separated values
        if current_value and isinstance(current_value[0], float):
            setattr(section_obj, field_name, [float(v.strip()) for v in value.split(",")])
        else:
            setattr(section_obj, field_name, [v.strip() for v in value.split(",")])
    else:
        setattr(section_obj, field_name, value)
