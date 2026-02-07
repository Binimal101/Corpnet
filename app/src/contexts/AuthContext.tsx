import { createContext, useContext, useEffect, useState, useCallback, type ReactNode } from 'react'
import type { User, Session } from '@supabase/supabase-js'
import { supabase } from '@/lib/supabase'
import type { Profile, OrgClaim, AuthContextType } from '@/types/auth'

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [org, setOrg] = useState<OrgClaim | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const extractClaims = useCallback((user: User) => {
    const appMeta = user.app_metadata || {}
    const orgClaim: OrgClaim | null = appMeta.organization || null
    setOrg(orgClaim)
  }, [])

  const fetchProfile = useCallback(async (userId: string) => {
    const { data } = await supabase
      .from('profiles')
      .select('*')
      .eq('id', userId)
      .single()

    if (data) setProfile(data as Profile)
  }, [])

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((event, session) => {
      setSession(session)
      setUser(session?.user ?? null)
      if (session?.user) {
        extractClaims(session.user)
        fetchProfile(session.user.id)
      } else {
        setProfile(null)
        setOrg(null)
      }
      if (event === 'INITIAL_SESSION' || event === 'SIGNED_IN' || event === 'SIGNED_OUT') {
        setIsLoading(false)
      }
    })

    return () => subscription.unsubscribe()
  }, [extractClaims, fetchProfile])

  const signInWithGoogle = useCallback(async (isSignup = false) => {
    if (isSignup) {
      localStorage.setItem('auth_flow', 'signup')
    } else {
      localStorage.setItem('auth_flow', 'login')
    }
    const redirectTo = `${window.location.origin}/auth/callback`
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo },
    })
  }, [])

  const signOut = useCallback(async () => {
    localStorage.removeItem('auth_flow')
    await supabase.auth.signOut()
    setUser(null)
    setSession(null)
    setProfile(null)
    setOrg(null)
  }, [])

  const refreshClaims = useCallback(async () => {
    const { data } = await supabase.auth.refreshSession()
    if (data.session?.user) {
      setSession(data.session)
      setUser(data.session.user)
      extractClaims(data.session.user)
    }
  }, [extractClaims])

  return (
    <AuthContext.Provider value={{
      user,
      profile,
      session,
      org,
      isLoading,
      signInWithGoogle,
      signOut,
      refreshClaims,
    }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
