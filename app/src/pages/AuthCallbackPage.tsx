import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { motion } from 'framer-motion'
import { AlertCircle, ArrowLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingScreen } from '@/components/LoadingScreen'
import { supabase } from '@/lib/supabase'

const AUTH_TIMEOUT_MS = 10_000

export function AuthCallbackPage() {
  const navigate = useNavigate()
  const [error, setError] = useState<string | null>(null)
  const handledRef = useRef(false)

  useEffect(() => {
    const queryError = new URLSearchParams(window.location.search).get('error_description')
    const hashError = new URLSearchParams(window.location.hash.substring(1)).get('error_description')

    if (queryError || hashError) {
      setError(decodeURIComponent(queryError || hashError || 'Authentication failed'))
      return
    }

    const authFlow = localStorage.getItem('auth_flow') || 'login'

    const handleSession = (session: { user: { app_metadata?: Record<string, unknown> } } | null) => {
      if (handledRef.current || !session) return false
      handledRef.current = true

      const orgClaim = session.user.app_metadata?.organization as Record<string, unknown> | null
      const isProduction = window.location.hostname === 'c0rpnet.tech'

      // Signup always goes to onboarding
      if (authFlow === 'signup') {
        localStorage.removeItem('auth_flow')
        if (orgClaim) {
          // Already has org (existing user clicked signup) → dashboard
          if (isProduction) {
            window.location.href = 'https://c0rpnet.tech/'
          } else {
            navigate('/', { replace: true })
          }
        } else {
          if (isProduction) {
            window.location.href = 'https://c0rpnet.tech/onboarding'
          } else {
            navigate('/onboarding', { replace: true })
          }
        }
        return true
      }

      // Login flow
      localStorage.removeItem('auth_flow')
      if (orgClaim) {
        if (isProduction) {
          window.location.href = 'https://c0rpnet.tech/'
        } else {
          navigate('/', { replace: true })
        }
      } else {
        if (isProduction) {
          window.location.href = 'https://c0rpnet.tech/onboarding'
        } else {
          navigate('/onboarding', { replace: true })
        }
      }
      return true
    }

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (handledRef.current) return

        if (event === 'SIGNED_IN') {
          handleSession(session)
          return
        }

        if (event === 'INITIAL_SESSION') {
          if (handleSession(session)) return

          if (!window.location.hash.includes('access_token')) {
            handledRef.current = true
            if (window.location.hostname === 'c0rpnet.tech') {
              window.location.href = 'https://c0rpnet.tech/login'
            } else {
              navigate('/login', { replace: true })
            }
          }
        }
      },
    )

    const timeout = setTimeout(() => {
      if (handledRef.current) return

      supabase.auth.getSession().then(({ data: { session } }) => {
        if (handledRef.current) return
        if (handleSession(session)) return

        handledRef.current = true
        setError('Authentication timed out. Please try again.')
      })
    }, AUTH_TIMEOUT_MS)

    return () => {
      subscription.unsubscribe()
      clearTimeout(timeout)
    }
  }, [navigate])

  if (error) {
    return (
      <div className="min-h-screen bg-background mosaic-bg flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="w-full max-w-md"
        >
          <div className="glass rounded-2xl border border-border/50 p-8 text-center">
            <div className="w-12 h-12 rounded-full bg-destructive/10 flex items-center justify-center mx-auto mb-4">
              <AlertCircle className="w-6 h-6 text-destructive" />
            </div>
            <h2 className="text-lg font-semibold text-foreground mb-2">Sign-in Failed</h2>
            <p className="text-sm text-muted-foreground mb-6">{error}</p>
            <Button
              onClick={() => {
                if (window.location.hostname === 'c0rpnet.tech') {
                  window.location.href = 'https://c0rpnet.tech/login'
                } else {
                  navigate('/login', { replace: true })
                }
              }}
              className="gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Login
            </Button>
          </div>
        </motion.div>
      </div>
    )
  }

  return <LoadingScreen />
}
