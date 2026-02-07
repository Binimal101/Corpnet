import { useState } from 'react'
import { useNavigate } from 'react-router'
import { Building2, Copy, Check, LogOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'

export function OrgSettingsPage() {
  const { org, user, refreshClaims } = useAuth()
  const navigate = useNavigate()
  const [orgName, setOrgName] = useState('')
  const [joinSecret, setJoinSecret] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [copied, setCopied] = useState(false)
  const [leaveError, setLeaveError] = useState('')
  const [isLeaving, setIsLeaving] = useState(false)

  // Load org details
  if (!loaded && org) {
    supabase
      .from('organizations')
      .select('*')
      .eq('id', org.org_id)
      .single()
      .then(({ data }) => {
        if (data) {
          setOrgName(data.name)
          setJoinSecret(data.join_secret || '')
        }
        setLoaded(true)
      })
  }

  const handleSave = async () => {
    if (!org || !orgName.trim()) return
    setIsLoading(true)

    await supabase
      .from('organizations')
      .update({ name: orgName.trim(), updated_at: new Date().toISOString() })
      .eq('id', org.org_id)

    await refreshClaims()
    setIsLoading(false)
  }

  const handleCopySecret = async () => {
    await navigator.clipboard.writeText(joinSecret)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleLeaveOrg = async () => {
    if (!org || !user) return
    setIsLeaving(true)
    setLeaveError('')

    const { error } = await supabase
      .from('organization_members')
      .delete()
      .eq('organization_id', org.org_id)
      .eq('user_id', user.id)

    if (error) {
      setLeaveError(error.message)
      setIsLeaving(false)
      return
    }

    await refreshClaims()
    navigate('/onboarding', { replace: true })
  }

  if (!org) return null

  return (
    <div className="space-y-6">
      <div className="glass rounded-xl border border-border/50 p-6">
        <div className="flex items-center gap-3 mb-6">
          <Building2 className="w-5 h-5 text-mosaic-purple" />
          <h2 className="text-lg font-semibold text-foreground">Organization Details</h2>
          {org.is_producer && (
            <Badge variant="outline" className="border-mosaic-purple text-mosaic-purple">
              Producer
            </Badge>
          )}
        </div>

        <div className="space-y-4 max-w-md">
          <div className="space-y-2">
            <Label htmlFor="name">Organization Name</Label>
            <Input
              id="name"
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
            />
          </div>

          <div className="space-y-2">
            <Label>Domain</Label>
            <Input value={org.org_domain} disabled />
            <p className="text-xs text-muted-foreground">Domain cannot be changed</p>
          </div>

          <div className="space-y-2">
            <Label>Slug</Label>
            <Input value={org.org_slug} disabled />
          </div>

          <div className="space-y-2">
            <Label>Your Role</Label>
            <Badge variant="outline">{org.role_name}</Badge>
          </div>

          {/* Invite Code */}
          <div className="space-y-2">
            <Label>Invite Code</Label>
            <div className="flex items-center gap-2">
              <Input
                value={joinSecret}
                readOnly
                className="font-mono tracking-wider"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={handleCopySecret}
                className="shrink-0"
              >
                {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Share this code with team members so they can join your organization
            </p>
          </div>

          {org.is_root && (
            <Button
              onClick={handleSave}
              disabled={isLoading}
              className="bg-gradient-to-r from-mosaic-purple to-mosaic-teal hover:opacity-90"
            >
              {isLoading ? 'Saving...' : 'Save Changes'}
            </Button>
          )}
        </div>
      </div>

      {/* Danger Zone */}
      <div className="glass rounded-xl border border-destructive/30 p-6">
        <h3 className="text-lg font-semibold text-destructive mb-1">Danger Zone</h3>
        <p className="text-sm text-muted-foreground mb-4">
          Leave this organization. You will need an invite code to rejoin.
        </p>

        {leaveError && (
          <div className="mb-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
            {leaveError}
          </div>
        )}

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button variant="destructive" className="gap-2">
              <LogOut className="w-4 h-4" />
              Leave Organization
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Leave organization?</AlertDialogTitle>
              <AlertDialogDescription>
                You will be removed from this organization and lose access to all its data.
                You will need an invite code to rejoin.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={handleLeaveOrg}
                disabled={isLeaving}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                {isLeaving ? 'Leaving...' : 'Leave'}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  )
}
