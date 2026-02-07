"""Supabase-backed permission management with hierarchical access.

Handles:
- Organization-level permissions from Supabase
- Hierarchical read access levels
- Access ID propagation to notes and chunks
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

log = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Hierarchical access levels (from most to least restrictive)."""
    ROOT = "root"           # Full access to everything
    ADMIN = "admin"         # Administrative access
    DEPARTMENT = "department"  # Department-level access
    TEAM = "team"           # Team-level access
    PROJECT = "project"     # Project-level access
    PUBLIC = "public"       # Public access (no restrictions)


@dataclass
class HierarchyNode:
    """A node in the access hierarchy.
    
    Attributes:
        level: The access level (root, admin, department, etc.)
        access_id: Unique identifier for this access scope
        name: Human-readable name
        parent_id: Parent node's access_id (None for root)
        children: Child node access_ids
    """
    level: AccessLevel
    access_id: str
    name: str
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "access_id": self.access_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "children": self.children,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HierarchyNode":
        return cls(
            level=AccessLevel(data["level"]),
            access_id=data["access_id"],
            name=data["name"],
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
        )


@dataclass
class ReadPermission:
    """A read permission with hierarchical access.
    
    Attributes:
        access_id: The access scope identifier
        level: Access level in hierarchy
        name: Human-readable name of the scope
        includes_children: Whether access includes child scopes
    """
    access_id: str
    level: AccessLevel
    name: str
    includes_children: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "access_id": self.access_id,
            "level": self.level.value,
            "name": self.name,
            "includes_children": self.includes_children,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReadPermission":
        return cls(
            access_id=data["access_id"],
            level=AccessLevel(data["level"]),
            name=data["name"],
            includes_children=data.get("includes_children", True),
        )


@dataclass
class OrganizationPermissions:
    """Permissions for an organization.
    
    Attributes:
        org_id: Organization identifier
        write_perm: Whether org has write access
        read_perms: List of hierarchical read permissions
        hierarchy: Full access hierarchy tree
    """
    org_id: str
    write_perm: bool
    read_perms: list[ReadPermission] = field(default_factory=list)
    hierarchy: dict[str, HierarchyNode] = field(default_factory=dict)
    
    def can_write(self) -> bool:
        return self.write_perm
    
    def can_read(self, access_id: str) -> bool:
        """Check if org can read content with given access_id."""
        for perm in self.read_perms:
            if perm.access_id == access_id:
                return True
            # Check if access_id is a child of a permitted scope
            if perm.includes_children and self._is_child_of(access_id, perm.access_id):
                return True
        return False
    
    def _is_child_of(self, child_id: str, parent_id: str) -> bool:
        """Check if child_id is a descendant of parent_id in the hierarchy."""
        if parent_id not in self.hierarchy:
            return False
        
        parent = self.hierarchy[parent_id]
        if child_id in parent.children:
            return True
        
        # Recursively check grandchildren
        for child in parent.children:
            if self._is_child_of(child_id, child):
                return True
        
        return False
    
    def get_accessible_ids(self) -> list[str]:
        """Get all access_ids this org can read."""
        accessible = set()
        
        for perm in self.read_perms:
            accessible.add(perm.access_id)
            if perm.includes_children:
                accessible.update(self._get_all_children(perm.access_id))
        
        return list(accessible)
    
    def _get_all_children(self, access_id: str) -> set[str]:
        """Recursively get all child access_ids."""
        children = set()
        
        if access_id not in self.hierarchy:
            return children
        
        node = self.hierarchy[access_id]
        for child_id in node.children:
            children.add(child_id)
            children.update(self._get_all_children(child_id))
        
        return children
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "org_id": self.org_id,
            "write_perm": self.write_perm,
            "read_perms": [p.to_dict() for p in self.read_perms],
            "accessible_ids": self.get_accessible_ids(),
        }


@dataclass
class UserPermissions:
    """User-level permissions inherited from organization.
    
    Attributes:
        user_id: User identifier
        org_id: Organization identifier
        org_permissions: Inherited organization permissions
        user_read_perms: User-specific read overrides (subset of org)
        user_write_perm: User-specific write override
    """
    user_id: str
    org_id: str
    org_permissions: OrganizationPermissions
    user_read_perms: list[ReadPermission] | None = None  # None = inherit all
    user_write_perm: bool | None = None  # None = inherit from org
    
    def can_write(self) -> bool:
        if self.user_write_perm is not None:
            return self.user_write_perm and self.org_permissions.write_perm
        return self.org_permissions.write_perm
    
    def can_read(self, access_id: str) -> bool:
        # User can only read what org allows
        if not self.org_permissions.can_read(access_id):
            return False
        
        # If user has specific read perms, check those
        if self.user_read_perms is not None:
            for perm in self.user_read_perms:
                if perm.access_id == access_id:
                    return True
                if perm.includes_children and self.org_permissions._is_child_of(access_id, perm.access_id):
                    return True
            return False
        
        # Inherit all org permissions
        return True
    
    def get_accessible_ids(self) -> list[str]:
        """Get all access_ids this user can read."""
        org_accessible = set(self.org_permissions.get_accessible_ids())
        
        if self.user_read_perms is None:
            return list(org_accessible)
        
        # Intersect with user-specific permissions
        user_accessible = set()
        for perm in self.user_read_perms:
            user_accessible.add(perm.access_id)
            if perm.includes_children:
                user_accessible.update(self.org_permissions._get_all_children(perm.access_id))
        
        return list(org_accessible & user_accessible)
    
    def get_default_access_id(self) -> str:
        """Get the default access_id for new content created by this user."""
        # Return the most specific (deepest) access level the user has
        accessible = self.get_accessible_ids()
        if not accessible:
            return "public"
        
        # Find the deepest node
        deepest = accessible[0]
        deepest_depth = 0
        
        for access_id in accessible:
            depth = self._get_depth(access_id)
            if depth > deepest_depth:
                deepest = access_id
                deepest_depth = depth
        
        return deepest
    
    def _get_depth(self, access_id: str) -> int:
        """Get depth of access_id in hierarchy (root = 0)."""
        depth = 0
        current = access_id
        
        while current in self.org_permissions.hierarchy:
            node = self.org_permissions.hierarchy[current]
            if node.parent_id is None:
                break
            current = node.parent_id
            depth += 1
        
        return depth
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "write_perm": self.can_write(),
            "read_perms": [p.to_dict() for p in (self.user_read_perms or self.org_permissions.read_perms)],
            "accessible_ids": self.get_accessible_ids(),
            "default_access_id": self.get_default_access_id(),
        }


class SupabasePermissionStore:
    """Supabase-backed permission storage.
    
    Expected Supabase schema:
    
    Table: organizations
        - org_id: text PRIMARY KEY
        - name: text
        - write_perm: boolean
        - created_at: timestamp
    
    Table: org_read_permissions
        - id: uuid PRIMARY KEY
        - org_id: text REFERENCES organizations(org_id)
        - access_id: text
        - level: text (root, admin, department, team, project, public)
        - name: text
        - includes_children: boolean
    
    Table: access_hierarchy
        - access_id: text PRIMARY KEY
        - org_id: text REFERENCES organizations(org_id)
        - level: text
        - name: text
        - parent_id: text REFERENCES access_hierarchy(access_id)
    
    Table: user_permissions (optional overrides)
        - user_id: text
        - org_id: text
        - write_perm: boolean (NULL = inherit)
        - PRIMARY KEY (user_id, org_id)
    
    Table: user_read_permissions (optional overrides)
        - user_id: text
        - org_id: text
        - access_id: text
        - includes_children: boolean
        - PRIMARY KEY (user_id, org_id, access_id)
    """
    
    def __init__(
        self,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
    ):
        """Initialize Supabase connection.
        
        Args:
            supabase_url: Supabase project URL. Defaults to SUPABASE_URL env var.
            supabase_key: Supabase API key. Defaults to SUPABASE_KEY env var.
        """
        self._url = supabase_url or os.getenv("SUPABASE_URL")
        self._key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self._url or not self._key:
            log.warning("Supabase credentials not configured. Using mock permissions.")
            self._client = None
        else:
            try:
                from supabase import create_client
                self._client = create_client(self._url, self._key)
                log.info("Connected to Supabase: %s", self._url)
            except ImportError:
                log.warning("supabase-py not installed. Using mock permissions.")
                self._client = None
            except Exception as e:
                log.error("Failed to connect to Supabase: %s", e)
                self._client = None
    
    def get_org_permissions(self, org_id: str) -> OrganizationPermissions | None:
        """Fetch organization permissions from Supabase.
        
        Args:
            org_id: Organization identifier.
            
        Returns:
            OrganizationPermissions or None if not found.
        """
        if self._client is None:
            return self._mock_org_permissions(org_id)
        
        try:
            # Fetch organization
            org_result = self._client.table("organizations").select("*").eq("org_id", org_id).execute()
            
            if not org_result.data:
                log.warning("Organization not found: %s", org_id)
                return None
            
            org_data = org_result.data[0]
            
            # Fetch read permissions
            read_result = self._client.table("org_read_permissions").select("*").eq("org_id", org_id).execute()
            
            read_perms = []
            for row in read_result.data:
                read_perms.append(ReadPermission(
                    access_id=row["access_id"],
                    level=AccessLevel(row["level"]),
                    name=row["name"],
                    includes_children=row.get("includes_children", True),
                ))
            
            # Fetch hierarchy
            hierarchy_result = self._client.table("access_hierarchy").select("*").eq("org_id", org_id).execute()
            
            hierarchy = {}
            for row in hierarchy_result.data:
                node = HierarchyNode(
                    level=AccessLevel(row["level"]),
                    access_id=row["access_id"],
                    name=row["name"],
                    parent_id=row.get("parent_id"),
                    children=[],
                )
                hierarchy[node.access_id] = node
            
            # Build parent-child relationships
            for node in hierarchy.values():
                if node.parent_id and node.parent_id in hierarchy:
                    hierarchy[node.parent_id].children.append(node.access_id)
            
            return OrganizationPermissions(
                org_id=org_id,
                write_perm=org_data.get("write_perm", False),
                read_perms=read_perms,
                hierarchy=hierarchy,
            )
            
        except Exception as e:
            log.error("Failed to fetch org permissions: %s", e)
            return None
    
    def get_user_permissions(
        self,
        user_id: str,
        org_id: str,
    ) -> UserPermissions | None:
        """Fetch user permissions (org + user-specific overrides).
        
        Args:
            user_id: User identifier.
            org_id: Organization identifier.
            
        Returns:
            UserPermissions or None if org not found.
        """
        # Get org permissions first
        org_perms = self.get_org_permissions(org_id)
        if org_perms is None:
            return None
        
        if self._client is None:
            return UserPermissions(
                user_id=user_id,
                org_id=org_id,
                org_permissions=org_perms,
            )
        
        try:
            # Check for user-specific overrides
            user_result = self._client.table("user_permissions").select("*").eq("user_id", user_id).eq("org_id", org_id).execute()
            
            user_write_perm = None
            if user_result.data:
                user_write_perm = user_result.data[0].get("write_perm")
            
            # Check for user-specific read permissions
            user_read_result = self._client.table("user_read_permissions").select("*").eq("user_id", user_id).eq("org_id", org_id).execute()
            
            user_read_perms = None
            if user_read_result.data:
                user_read_perms = []
                for row in user_read_result.data:
                    # Find the matching org perm to get level and name
                    access_id = row["access_id"]
                    level = AccessLevel.PUBLIC
                    name = access_id
                    
                    for org_perm in org_perms.read_perms:
                        if org_perm.access_id == access_id:
                            level = org_perm.level
                            name = org_perm.name
                            break
                    
                    user_read_perms.append(ReadPermission(
                        access_id=access_id,
                        level=level,
                        name=name,
                        includes_children=row.get("includes_children", True),
                    ))
            
            return UserPermissions(
                user_id=user_id,
                org_id=org_id,
                org_permissions=org_perms,
                user_read_perms=user_read_perms,
                user_write_perm=user_write_perm,
            )
            
        except Exception as e:
            log.error("Failed to fetch user permissions: %s", e)
            return UserPermissions(
                user_id=user_id,
                org_id=org_id,
                org_permissions=org_perms,
            )
    
    def _mock_org_permissions(self, org_id: str) -> OrganizationPermissions:
        """Return mock permissions for testing without Supabase."""
        log.info("Using mock permissions for org: %s", org_id)
        
        # Create a sample hierarchy
        hierarchy = {
            "root": HierarchyNode(
                level=AccessLevel.ROOT,
                access_id="root",
                name="Root",
                parent_id=None,
                children=["dept_engineering", "dept_sales"],
            ),
            "dept_engineering": HierarchyNode(
                level=AccessLevel.DEPARTMENT,
                access_id="dept_engineering",
                name="Engineering",
                parent_id="root",
                children=["team_backend", "team_frontend"],
            ),
            "dept_sales": HierarchyNode(
                level=AccessLevel.DEPARTMENT,
                access_id="dept_sales",
                name="Sales",
                parent_id="root",
                children=["team_enterprise", "team_smb"],
            ),
            "team_backend": HierarchyNode(
                level=AccessLevel.TEAM,
                access_id="team_backend",
                name="Backend Team",
                parent_id="dept_engineering",
                children=["proj_api", "proj_db"],
            ),
            "team_frontend": HierarchyNode(
                level=AccessLevel.TEAM,
                access_id="team_frontend",
                name="Frontend Team",
                parent_id="dept_engineering",
                children=["proj_web", "proj_mobile"],
            ),
            "team_enterprise": HierarchyNode(
                level=AccessLevel.TEAM,
                access_id="team_enterprise",
                name="Enterprise Sales",
                parent_id="dept_sales",
                children=[],
            ),
            "team_smb": HierarchyNode(
                level=AccessLevel.TEAM,
                access_id="team_smb",
                name="SMB Sales",
                parent_id="dept_sales",
                children=[],
            ),
            "proj_api": HierarchyNode(
                level=AccessLevel.PROJECT,
                access_id="proj_api",
                name="API Project",
                parent_id="team_backend",
                children=[],
            ),
            "proj_db": HierarchyNode(
                level=AccessLevel.PROJECT,
                access_id="proj_db",
                name="Database Project",
                parent_id="team_backend",
                children=[],
            ),
            "proj_web": HierarchyNode(
                level=AccessLevel.PROJECT,
                access_id="proj_web",
                name="Web App",
                parent_id="team_frontend",
                children=[],
            ),
            "proj_mobile": HierarchyNode(
                level=AccessLevel.PROJECT,
                access_id="proj_mobile",
                name="Mobile App",
                parent_id="team_frontend",
                children=[],
            ),
        }
        
        # Mock read permissions - org has access to engineering department
        read_perms = [
            ReadPermission(
                access_id="dept_engineering",
                level=AccessLevel.DEPARTMENT,
                name="Engineering",
                includes_children=True,
            ),
        ]
        
        return OrganizationPermissions(
            org_id=org_id,
            write_perm=True,
            read_perms=read_perms,
            hierarchy=hierarchy,
        )
