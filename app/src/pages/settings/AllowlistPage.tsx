import { useState, useEffect } from 'react'
import { Shield, Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'
import type { AllowedDomain } from '@/types/auth'

export function AllowlistPage() {
  const { org: activeOrg, user } = useAuth()
  const [domains, setDomains] = useState<AllowedDomain[]>([])
  const [newDomain, setNewDomain] = useState('')
  const [isLoading, setIsLoading] = useState(true)
  const [isAdding, setIsAdding] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!activeOrg) return

    const fetchDomains = async () => {
      const { data } = await supabase
        .from('allowed_domains')
        .select('*')
        .eq('producer_org_id', activeOrg.org_id)
        .order('created_at', { ascending: true })

      setDomains((data as AllowedDomain[]) || [])
      setIsLoading(false)
    }

    fetchDomains()
  }, [activeOrg])

  const handleAddDomain = async () => {
    if (!activeOrg || !user || !newDomain.trim()) return
    setIsAdding(true)
    setError('')

    const { data, error: insertError } = await supabase
      .from('allowed_domains')
      .insert({
        producer_org_id: activeOrg.org_id,
        allowed_domain: newDomain.trim().toLowerCase(),
        created_by: user.id,
      })
      .select()
      .single()

    if (insertError) {
      setError(insertError.message)
      setIsAdding(false)
      return
    }

    setDomains(prev => [...prev, data as AllowedDomain])
    setNewDomain('')
    setIsAdding(false)
  }

  const handleRemoveDomain = async (domainId: string) => {
    await supabase
      .from('allowed_domains')
      .delete()
      .eq('id', domainId)

    setDomains(prev => prev.filter(d => d.id !== domainId))
  }

  if (!activeOrg?.is_producer) {
    return (
      <div className="glass rounded-xl border border-border/50 p-6 text-center">
        <Shield className="w-8 h-8 text-muted-foreground mx-auto mb-3" />
        <p className="text-muted-foreground">
          Domain allowlist is only available for producer organizations.
        </p>
      </div>
    )
  }

  const isOwnerOrAdmin = activeOrg.is_root

  return (
    <div className="space-y-6">
      <div className="glass rounded-xl border border-border/50 p-6">
        <div className="flex items-center gap-3 mb-2">
          <Shield className="w-5 h-5 text-mosaic-purple" />
          <h2 className="text-lg font-semibold text-foreground">Domain Allowlist</h2>
          <Badge variant="outline">{domains.length}</Badge>
        </div>
        <p className="text-sm text-muted-foreground mb-6">
          Consumer organizations with these domains can access your data.
        </p>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}

        {/* Add domain */}
        {isOwnerOrAdmin && (
          <div className="flex gap-2 mb-6">
            <Input
              placeholder="e.g. consumer.com"
              value={newDomain}
              onChange={(e) => setNewDomain(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAddDomain()}
            />
            <Button
              onClick={handleAddDomain}
              disabled={isAdding || !newDomain.trim()}
              className="gap-2 bg-gradient-to-r from-mosaic-purple to-mosaic-teal hover:opacity-90"
            >
              <Plus className="w-4 h-4" />
              Add
            </Button>
          </div>
        )}

        {/* Domain list */}
        {isLoading ? (
          <div className="text-center py-8 text-muted-foreground text-sm">Loading domains...</div>
        ) : domains.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground text-sm">
            No domains in the allowlist yet. Add consumer domains to grant access.
          </div>
        ) : (
          <div className="space-y-2">
            {domains.map((domain) => (
              <div
                key={domain.id}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-mosaic-teal" />
                  <span className="text-sm font-medium text-foreground">{domain.allowed_domain}</span>
                </div>
                {isOwnerOrAdmin && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => handleRemoveDomain(domain.id)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
