import { NavLink, Outlet } from 'react-router'
import { Settings, Users, Shield, FolderTree } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useAuth } from '@/contexts/AuthContext'

const navItems = [
  { to: '/settings', label: 'General', icon: Settings, end: true },
  { to: '/settings/members', label: 'Members', icon: Users, end: false },
  { to: '/settings/roles', label: 'Roles', icon: FolderTree, end: false },
  { to: '/settings/allowlist', label: 'Domain Allowlist', icon: Shield, end: false, producerOnly: true },
]

export function SettingsLayout() {
  const { org } = useAuth()

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage your organization settings
        </p>
      </div>

      <div className="flex gap-6">
        <nav className="w-48 flex-shrink-0 space-y-1">
          {navItems
            .filter(item => !item.producerOnly || org?.is_producer)
            .map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors',
                    isActive
                      ? 'bg-mosaic-purple/20 text-mosaic-purple'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                  )
                }
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </NavLink>
            ))}
        </nav>

        <div className="flex-1 min-w-0">
          <Outlet />
        </div>
      </div>
    </div>
  )
}
