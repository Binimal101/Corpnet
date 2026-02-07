import { useState } from 'react'
import { Outlet } from 'react-router'
import { motion } from 'framer-motion'
import {
  Network,
  Search,
  Database,
  FileText,
  Settings,
  HelpCircle,
  Menu,
  X,
  ChevronRight,
  Building2,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { UserMenu } from '@/components/auth/UserMenu'
import { useAuth } from '@/contexts/AuthContext'

export function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { org } = useAuth()

  return (
    <div className="min-h-screen bg-background mosaic-bg">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="fixed top-0 left-0 right-0 z-40 glass border-b border-border/50"
      >
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>

            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-mosaic-purple to-mosaic-teal flex items-center justify-center">
                <Network className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-lg bg-gradient-to-r from-mosaic-purple via-mosaic-teal to-mosaic-coral bg-clip-text text-transparent">
                Mosaic RAG
              </span>
            </div>

            {org && (
              <div className="hidden sm:flex items-center gap-2 border-l border-border/50 pl-3 ml-1">
                <Building2 className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium">{org.org_domain}</span>
                {org.is_producer && (
                  <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-mosaic-purple text-mosaic-purple">
                    Producer
                  </Badge>
                )}
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="hidden sm:flex">
              <HelpCircle className="w-5 h-5" />
            </Button>
            <Button variant="ghost" size="icon" className="hidden sm:flex">
              <Settings className="w-5 h-5" />
            </Button>
            <UserMenu />
          </div>
        </div>
      </motion.header>

      {/* Sidebar */}
      <motion.aside
        initial={{ x: -300 }}
        animate={{ x: sidebarOpen ? 0 : -300 }}
        transition={{ type: 'spring', damping: 25 }}
        className="fixed left-0 top-16 bottom-0 w-64 glass border-r border-border/50 z-30 lg:translate-x-0"
      >
        <div className="p-4 space-y-6">
          {/* Stats */}
          <div className="space-y-3">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Knowledge Base
            </h3>
            <div className="grid grid-cols-2 gap-2">
              <div className="p-3 rounded-xl bg-muted/50">
                <div className="text-2xl font-bold text-mosaic-purple">2.4k</div>
                <div className="text-xs text-muted-foreground">Documents</div>
              </div>
              <div className="p-3 rounded-xl bg-muted/50">
                <div className="text-2xl font-bold text-mosaic-teal">15k</div>
                <div className="text-xs text-muted-foreground">Entities</div>
              </div>
              <div className="p-3 rounded-xl bg-muted/50">
                <div className="text-2xl font-bold text-mosaic-coral">8.2k</div>
                <div className="text-xs text-muted-foreground">Concepts</div>
              </div>
              <div className="p-3 rounded-xl bg-muted/50">
                <div className="text-2xl font-bold text-mosaic-gold">45k</div>
                <div className="text-xs text-muted-foreground">Connections</div>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="space-y-2">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Navigation
            </h3>
            <nav className="space-y-1">
              {[
                { icon: Search, label: 'Search', active: true },
                { icon: Database, label: 'Documents', active: false },
                { icon: Network, label: 'Graph View', active: false },
                { icon: FileText, label: 'Sources', active: false },
              ].map((item) => (
                <button
                  key={item.label}
                  className={`
                    w-full flex items-center gap-3 px-3 py-2 rounded-lg
                    transition-colors
                    ${item.active
                      ? 'bg-mosaic-purple/20 text-mosaic-purple'
                      : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    }
                  `}
                >
                  <item.icon className="w-4 h-4" />
                  {item.label}
                </button>
              ))}
            </nav>
          </div>

          {/* Recent searches */}
          <div className="space-y-2">
            <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Recent
            </h3>
            <div className="space-y-1">
              {[
                'Machine learning basics',
                'Neural network types',
                'Transformer architecture',
              ].map((search) => (
                <button
                  key={search}
                  className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors text-left"
                >
                  <ChevronRight className="w-3 h-3" />
                  {search}
                </button>
              ))}
            </div>
          </div>
        </div>
      </motion.aside>

      {/* Main content */}
      <main className="pt-16 lg:pl-64 min-h-screen">
        <div className="max-w-5xl mx-auto px-4 py-8 ml-8">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
