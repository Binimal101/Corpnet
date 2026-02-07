import { createBrowserRouter } from 'react-router'
import { ProtectedRoute } from '@/components/auth/ProtectedRoute'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { LoginPage } from '@/pages/LoginPage'
import { SignupPage } from '@/pages/SignupPage'
import { AuthCallbackPage } from '@/pages/AuthCallbackPage'
import { OnboardingPage } from '@/pages/OnboardingPage'
import { DashboardPage } from '@/pages/DashboardPage'
import { SettingsLayout } from '@/pages/settings/SettingsLayout'
import { OrgSettingsPage } from '@/pages/settings/OrgSettingsPage'
import { MembersPage } from '@/pages/settings/MembersPage'
import { RolesPage } from '@/pages/settings/RolesPage'
import { AllowlistPage } from '@/pages/settings/AllowlistPage'

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <LoginPage />,
  },
  {
    path: '/signup',
    element: <SignupPage />,
  },
  {
    path: '/auth/callback',
    element: <AuthCallbackPage />,
  },
  {
    path: '/onboarding',
    element: (
      <ProtectedRoute requireOrg={false}>
        <OnboardingPage />
      </ProtectedRoute>
    ),
  },
  {
    path: '/',
    element: (
      <ProtectedRoute>
        <DashboardLayout />
      </ProtectedRoute>
    ),
    children: [
      { index: true, element: <DashboardPage /> },
      {
        path: 'settings',
        element: <SettingsLayout />,
        children: [
          { index: true, element: <OrgSettingsPage /> },
          { path: 'members', element: <MembersPage /> },
          { path: 'roles', element: <RolesPage /> },
          { path: 'allowlist', element: <AllowlistPage /> },
        ],
      },
    ],
  },
])
