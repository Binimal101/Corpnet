import { Navigate } from 'react-router'
import { useAuth } from '@/contexts/AuthContext'
import { LoadingScreen } from '@/components/LoadingScreen'

interface ProtectedRouteProps {
  children: React.ReactNode
  requireOrg?: boolean
}

export function ProtectedRoute({ children, requireOrg = true }: ProtectedRouteProps) {
  const { user, org, isLoading } = useAuth()

  if (isLoading) {
    return <LoadingScreen />
  }

  if (!user) {
    return <Navigate to="/login" replace />
  }

  if (requireOrg && !org) {
    return <Navigate to="/onboarding" replace />
  }

  return <>{children}</>
}
