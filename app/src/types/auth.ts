import type { User, Session } from '@supabase/supabase-js'

export interface OrgClaim {
  org_id: string
  org_slug: string
  org_domain: string
  role_id: string
  role_name: string
  is_root: boolean
  is_producer: boolean
}

export interface Profile {
  id: string
  email: string
  full_name: string
  avatar_url: string
  created_at: string
  updated_at: string
}

export interface Organization {
  id: string
  name: string
  slug: string
  domain: string
  is_producer: boolean
  join_secret: string
  created_by: string
  created_at: string
  updated_at: string
}

export interface Role {
  id: string
  organization_id: string
  name: string
  parent_role_id: string | null
  created_at: string
}

export interface OrganizationMember {
  id: string
  organization_id: string
  user_id: string
  role_id: string
  created_at: string
  profiles?: Profile
  roles?: Role
}

export interface AllowedDomain {
  id: string
  producer_org_id: string
  allowed_domain: string
  created_by: string
  created_at: string
}

export interface AuthContextType {
  user: User | null
  profile: Profile | null
  session: Session | null
  org: OrgClaim | null
  isLoading: boolean
  signInWithGoogle: (isSignup?: boolean) => Promise<void>
  signOut: () => Promise<void>
  refreshClaims: () => Promise<void>
}
