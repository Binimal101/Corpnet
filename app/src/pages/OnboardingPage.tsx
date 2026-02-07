import { useState } from 'react'
import { useNavigate } from 'react-router'
import { motion } from 'framer-motion'
import { Building2, Plus, Users, Network, Copy, Check, KeyRound } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'
import type { Organization } from '@/types/auth'

export function OnboardingPage() {
  const { user, refreshClaims } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<'choose' | 'create' | 'join'>('choose')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState('')

  // Create org form
  const [orgName, setOrgName] = useState('')
  const [orgDomain, setOrgDomain] = useState('')

  // After creation - show join secret
  const [createdOrg, setCreatedOrg] = useState<Organization | null>(null)
  const [copied, setCopied] = useState(false)

  // Join org
  const [joinCode, setJoinCode] = useState('')
  const [foundOrg, setFoundOrg] = useState<Organization | null>(null)
  const [lookingUp, setLookingUp] = useState(false)

  const handleCreateOrg = async () => {
    if (!orgName.trim() || !orgDomain.trim() || !user) return
    setIsSubmitting(true)
    setError('')

    const slug = orgName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')

    const { data: org, error: orgError } = await supabase
      .from('organizations')
      .insert({
        name: orgName.trim(),
        slug,
        domain: orgDomain.trim().toLowerCase(),
        created_by: user.id,
      })
      .select()
      .single()

    if (orgError) {
      setError(orgError.message)
      setIsSubmitting(false)
      return
    }

    // The DB trigger auto_create_root_role() already created the Owner role
    // and added the creator as a member, so just refresh claims
    await refreshClaims()
    setCreatedOrg(org as Organization)
    setIsSubmitting(false)
  }

  const handleLookupCode = async () => {
    if (!joinCode.trim()) return
    setLookingUp(true)
    setError('')
    setFoundOrg(null)

    const { data, error: lookupError } = await supabase
      .from('organizations')
      .select('*')
      .eq('join_secret', joinCode.trim().toUpperCase())
      .single()

    if (lookupError || !data) {
      setError('Invalid invite code. Please check and try again.')
      setLookingUp(false)
      return
    }

    setFoundOrg(data as Organization)
    setLookingUp(false)
  }

  const handleJoinOrg = async () => {
    if (!user || !foundOrg) return
    setIsSubmitting(true)
    setError('')

    const { error: joinError } = await supabase
      .rpc('join_org_by_secret', { secret: foundOrg.join_secret })

    if (joinError) {
      setError(joinError.message)
      setIsSubmitting(false)
      return
    }

    await refreshClaims()
    navigate('/', { replace: true })
  }

  const handleCopySecret = async () => {
    if (!createdOrg?.join_secret) return
    await navigator.clipboard.writeText(createdOrg.join_secret)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // After org creation, show the join secret
  if (createdOrg) {
    return (
      <div className="min-h-screen bg-background mosaic-bg flex items-center justify-center px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-lg"
        >
          <div className="glass rounded-2xl border border-border/50 p-8">
            <div className="text-center mb-6">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-mosaic-purple to-mosaic-teal flex items-center justify-center mx-auto mb-4">
                <Check className="w-6 h-6 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-foreground">Organization Created!</h2>
              <p className="text-sm text-muted-foreground mt-1">
                Share this invite code with your team members
              </p>
            </div>

            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-muted/50 border border-border/50">
                <Label className="text-xs text-muted-foreground uppercase tracking-wider">Invite Code</Label>
                <div className="flex items-center gap-2 mt-2">
                  <code className="flex-1 text-2xl font-mono font-bold tracking-widest text-foreground">
                    {createdOrg.join_secret}
                  </code>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={handleCopySecret}
                    className="shrink-0"
                  >
                    {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
                  </Button>
                </div>
              </div>

              <Button
                onClick={() => navigate('/', { replace: true })}
                className="w-full bg-gradient-to-r from-mosaic-purple to-mosaic-teal hover:opacity-90"
              >
                Go to Dashboard
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background mosaic-bg flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-lg"
      >
        <div className="glass rounded-2xl border border-border/50 p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center gap-2 mb-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-mosaic-purple to-mosaic-teal flex items-center justify-center">
                <Network className="w-4 h-4 text-white" />
              </div>
              <span className="font-bold text-xl bg-gradient-to-r from-mosaic-purple via-mosaic-teal to-mosaic-coral bg-clip-text text-transparent">
                CorpNet
              </span>
            </div>
            <h2 className="text-2xl font-bold text-foreground mt-4">
              {mode === 'choose' && 'Get Started'}
              {mode === 'create' && 'Create Organization'}
              {mode === 'join' && 'Join Organization'}
            </h2>
            <p className="text-sm text-muted-foreground mt-1">
              {mode === 'choose' && 'Create or join an organization to continue'}
              {mode === 'create' && 'Set up a new organization'}
              {mode === 'join' && 'Enter an invite code to join'}
            </p>
          </div>

          {error && (
            <div className="mb-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
              {error}
            </div>
          )}

          {/* Choose mode */}
          {mode === 'choose' && (
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setMode('create')}
                className="p-6 rounded-xl border border-border/50 hover:border-mosaic-purple/30 bg-muted/30 hover:bg-muted/50 transition-all text-left"
              >
                <Plus className="w-8 h-8 text-mosaic-purple mb-3" />
                <h3 className="font-semibold text-foreground">Create</h3>
                <p className="text-xs text-muted-foreground mt-1">Start a new organization</p>
              </button>
              <button
                onClick={() => setMode('join')}
                className="p-6 rounded-xl border border-border/50 hover:border-mosaic-teal/30 bg-muted/30 hover:bg-muted/50 transition-all text-left"
              >
                <Users className="w-8 h-8 text-mosaic-teal mb-3" />
                <h3 className="font-semibold text-foreground">Join</h3>
                <p className="text-xs text-muted-foreground mt-1">Use an invite code</p>
              </button>
            </div>
          )}

          {/* Create org form */}
          {mode === 'create' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="orgName">Organization Name</Label>
                <Input
                  id="orgName"
                  placeholder="e.g. Acme Corp"
                  value={orgName}
                  onChange={(e) => setOrgName(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="orgDomain">Domain</Label>
                <Input
                  id="orgDomain"
                  placeholder="e.g. acme.com"
                  value={orgDomain}
                  onChange={(e) => setOrgDomain(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Your organization's email domain
                </p>
              </div>
              <div className="flex gap-3 pt-2">
                <Button variant="outline" onClick={() => { setMode('choose'); setError('') }} className="flex-1">
                  Back
                </Button>
                <Button
                  onClick={handleCreateOrg}
                  disabled={isSubmitting || !orgName.trim() || !orgDomain.trim()}
                  className="flex-1 bg-gradient-to-r from-mosaic-purple to-mosaic-teal hover:opacity-90"
                >
                  {isSubmitting ? 'Creating...' : 'Create'}
                </Button>
              </div>
            </div>
          )}

          {/* Join org */}
          {mode === 'join' && (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="joinCode">Invite Code</Label>
                <div className="flex gap-2">
                  <Input
                    id="joinCode"
                    placeholder="e.g. A1B2C3D4E5F6"
                    value={joinCode}
                    onChange={(e) => {
                      setJoinCode(e.target.value.toUpperCase())
                      setFoundOrg(null)
                      setError('')
                    }}
                    className="font-mono tracking-wider"
                  />
                  <Button
                    variant="outline"
                    onClick={handleLookupCode}
                    disabled={lookingUp || !joinCode.trim()}
                  >
                    <KeyRound className="w-4 h-4" />
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Get this code from your organization admin
                </p>
              </div>

              {foundOrg && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <button
                    onClick={handleJoinOrg}
                    disabled={isSubmitting}
                    className="w-full p-4 rounded-xl border border-mosaic-teal/30 bg-mosaic-teal/5 hover:bg-mosaic-teal/10 transition-all text-left flex items-center gap-3"
                  >
                    <Building2 className="w-5 h-5 text-mosaic-teal" />
                    <div className="flex-1">
                      <div className="font-medium text-foreground">{foundOrg.name}</div>
                      <div className="text-xs text-muted-foreground">{foundOrg.domain}</div>
                    </div>
                    <span className="text-sm text-mosaic-teal font-medium">
                      {isSubmitting ? 'Joining...' : 'Join'}
                    </span>
                  </button>
                </motion.div>
              )}

              <div className="flex gap-3 pt-2">
                <Button variant="outline" onClick={() => { setMode('choose'); setError(''); setFoundOrg(null); setJoinCode('') }} className="flex-1">
                  Back
                </Button>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  )
}
