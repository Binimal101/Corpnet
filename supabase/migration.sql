-- ============================================
-- Corpnet: Auth + Producer/Consumer Org System
-- Run this in Supabase SQL Editor
-- ============================================

-- 1. TABLES
-- ---------

-- Profiles (extends auth.users)
CREATE TABLE public.profiles (
  id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email text NOT NULL,
  full_name text DEFAULT '',
  avatar_url text DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Organizations
CREATE TABLE public.organizations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  slug text NOT NULL UNIQUE,
  domain text NOT NULL UNIQUE,
  is_producer boolean NOT NULL DEFAULT false,
  created_by uuid NOT NULL REFERENCES public.profiles(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Organization Members
CREATE TABLE public.organization_members (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  organization_id uuid NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  role text NOT NULL CHECK (role IN ('owner', 'admin', 'member')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (organization_id, user_id)
);

-- Allowed Domains (producer's consumer allowlist)
CREATE TABLE public.allowed_domains (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  producer_org_id uuid NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
  allowed_domain text NOT NULL,
  created_by uuid NOT NULL REFERENCES public.profiles(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (producer_org_id, allowed_domain)
);

-- 2. FUNCTIONS & TRIGGERS
-- -----------------------

-- Auto-create profile on user sign-up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name, avatar_url)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data ->> 'full_name', NEW.raw_user_meta_data ->> 'name', ''),
    COALESCE(NEW.raw_user_meta_data ->> 'avatar_url', NEW.raw_user_meta_data ->> 'picture', '')
  );
  RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Update JWT claims when membership changes
CREATE OR REPLACE FUNCTION public.update_user_claims(target_user_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
DECLARE
  org_claims jsonb;
  current_active uuid;
BEGIN
  SELECT COALESCE(jsonb_agg(
    jsonb_build_object(
      'org_id', om.organization_id,
      'org_slug', o.slug,
      'org_domain', o.domain,
      'role', om.role,
      'is_producer', o.is_producer
    )
  ), '[]'::jsonb)
  INTO org_claims
  FROM public.organization_members om
  JOIN public.organizations o ON o.id = om.organization_id
  WHERE om.user_id = target_user_id;

  SELECT (auth.users.raw_app_meta_data ->> 'active_org_id')::uuid
  INTO current_active
  FROM auth.users
  WHERE id = target_user_id;

  IF current_active IS NULL OR NOT EXISTS (
    SELECT 1 FROM public.organization_members
    WHERE user_id = target_user_id AND organization_id = current_active
  ) THEN
    SELECT (org_claims -> 0 ->> 'org_id')::uuid INTO current_active;
  END IF;

  UPDATE auth.users
  SET raw_app_meta_data = COALESCE(raw_app_meta_data, '{}'::jsonb)
    || jsonb_build_object('organizations', org_claims)
    || jsonb_build_object('active_org_id', current_active)
  WHERE id = target_user_id;
END;
$$;

-- Trigger on membership changes
CREATE OR REPLACE FUNCTION public.on_membership_change()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    PERFORM public.update_user_claims(OLD.user_id);
    RETURN OLD;
  ELSE
    PERFORM public.update_user_claims(NEW.user_id);
    RETURN NEW;
  END IF;
END;
$$;

CREATE TRIGGER on_org_member_change
  AFTER INSERT OR UPDATE OR DELETE ON public.organization_members
  FOR EACH ROW EXECUTE FUNCTION public.on_membership_change();

-- RPC: Switch active org
CREATE OR REPLACE FUNCTION public.set_active_org(target_org_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = ''
AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM public.organization_members
    WHERE user_id = auth.uid() AND organization_id = target_org_id
  ) THEN
    RAISE EXCEPTION 'User is not a member of this organization';
  END IF;

  UPDATE auth.users
  SET raw_app_meta_data = COALESCE(raw_app_meta_data, '{}'::jsonb)
    || jsonb_build_object('active_org_id', target_org_id)
  WHERE id = auth.uid();
END;
$$;

-- Prevent removing the last owner
CREATE OR REPLACE FUNCTION public.prevent_last_owner_removal()
RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.role = 'owner' AND (
    SELECT COUNT(*) FROM public.organization_members
    WHERE organization_id = OLD.organization_id AND role = 'owner'
  ) <= 1 THEN
    RAISE EXCEPTION 'Cannot remove the last owner of an organization';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER prevent_last_owner
  BEFORE DELETE ON public.organization_members
  FOR EACH ROW EXECUTE FUNCTION public.prevent_last_owner_removal();

-- 3. ROW LEVEL SECURITY
-- ---------------------

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.allowed_domains ENABLE ROW LEVEL SECURITY;

-- Profiles
CREATE POLICY "Users can read own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can read org co-members"
  ON public.profiles FOR SELECT
  USING (
    id IN (
      SELECT om.user_id FROM public.organization_members om
      WHERE om.organization_id IN (
        SELECT om2.organization_id FROM public.organization_members om2
        WHERE om2.user_id = auth.uid()
      )
    )
  );

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id)
  WITH CHECK (auth.uid() = id);

-- Organizations
CREATE POLICY "Members can read own orgs"
  ON public.organizations FOR SELECT
  USING (
    id IN (
      SELECT organization_id FROM public.organization_members
      WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Anyone can read orgs for domain matching"
  ON public.organizations FOR SELECT
  USING (auth.uid() IS NOT NULL);

CREATE POLICY "Authenticated users can create orgs"
  ON public.organizations FOR INSERT
  WITH CHECK (auth.uid() = created_by);

CREATE POLICY "Owners and admins can update org"
  ON public.organizations FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.organization_members
      WHERE organization_id = organizations.id
        AND user_id = auth.uid()
        AND role IN ('owner', 'admin')
    )
  );

-- Organization Members
CREATE POLICY "Members can read org members"
  ON public.organization_members FOR SELECT
  USING (
    organization_id IN (
      SELECT organization_id FROM public.organization_members AS om
      WHERE om.user_id = auth.uid()
    )
  );

CREATE POLICY "Owners/admins can add members or self-join"
  ON public.organization_members FOR INSERT
  WITH CHECK (
    user_id = auth.uid()
    OR
    EXISTS (
      SELECT 1 FROM public.organization_members om
      WHERE om.organization_id = organization_members.organization_id
        AND om.user_id = auth.uid()
        AND om.role IN ('owner', 'admin')
    )
  );

CREATE POLICY "Owners can remove members or self-leave"
  ON public.organization_members FOR DELETE
  USING (
    user_id = auth.uid()
    OR
    EXISTS (
      SELECT 1 FROM public.organization_members AS om
      WHERE om.organization_id = organization_members.organization_id
        AND om.user_id = auth.uid()
        AND om.role = 'owner'
    )
  );

CREATE POLICY "Owners/admins can update member roles"
  ON public.organization_members FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.organization_members AS om
      WHERE om.organization_id = organization_members.organization_id
        AND om.user_id = auth.uid()
        AND om.role IN ('owner', 'admin')
    )
  );

-- Allowed Domains
CREATE POLICY "Producer members can read allowlist"
  ON public.allowed_domains FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.organization_members
      WHERE organization_id = allowed_domains.producer_org_id
        AND user_id = auth.uid()
    )
  );

CREATE POLICY "Producer owners/admins can insert domains"
  ON public.allowed_domains FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.organization_members
      WHERE organization_id = allowed_domains.producer_org_id
        AND user_id = auth.uid()
        AND role IN ('owner', 'admin')
    )
  );

CREATE POLICY "Producer owners/admins can delete domains"
  ON public.allowed_domains FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM public.organization_members
      WHERE organization_id = allowed_domains.producer_org_id
        AND user_id = auth.uid()
        AND role IN ('owner', 'admin')
    )
  );
