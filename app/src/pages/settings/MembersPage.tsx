import { useState, useEffect } from 'react'
import { Users, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'
import type { Role } from '@/types/auth'

interface Member {
  id: string
  user_id: string
  role_id: string
  created_at: string
  profiles: {
    full_name: string
    email: string
    avatar_url: string
  }
  roles: {
    id: string
    name: string
    parent_role_id: string | null
  }
}

export function MembersPage() {
  const { org, user } = useAuth()
  const [members, setMembers] = useState<Member[]>([])
  const [roles, setRoles] = useState<Role[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    if (!org) return

    const fetchData = async () => {
      const [{ data: membersData }, { data: rolesData }] = await Promise.all([
        supabase
          .from('organization_members')
          .select('id, user_id, role_id, created_at, profiles(full_name, email, avatar_url), roles(id, name, parent_role_id)')
          .eq('organization_id', org.org_id)
          .order('created_at', { ascending: true }),
        supabase
          .from('roles')
          .select('*')
          .eq('organization_id', org.org_id)
          .order('created_at', { ascending: true }),
      ])

      setMembers((membersData as unknown as Member[]) || [])
      setRoles((rolesData as Role[]) || [])
      setIsLoading(false)
    }

    fetchData()
  }, [org])

  const handleRemoveMember = async (memberId: string) => {
    await supabase
      .from('organization_members')
      .delete()
      .eq('id', memberId)

    setMembers(prev => prev.filter(m => m.id !== memberId))
  }

  const handleRoleChange = async (memberId: string, newRoleId: string) => {
    const { error } = await supabase
      .from('organization_members')
      .update({ role_id: newRoleId })
      .eq('id', memberId)

    if (!error) {
      setMembers(prev => prev.map(m => {
        if (m.id !== memberId) return m
        const newRole = roles.find(r => r.id === newRoleId)
        return {
          ...m,
          role_id: newRoleId,
          roles: newRole ? { id: newRole.id, name: newRole.name, parent_role_id: newRole.parent_role_id } : m.roles,
        }
      }))
    }
  }

  const canManage = org?.is_root ?? false

  return (
    <div className="space-y-6">
      <div className="glass rounded-xl border border-border/50 p-6">
        <div className="flex items-center gap-3 mb-6">
          <Users className="w-5 h-5 text-mosaic-teal" />
          <h2 className="text-lg font-semibold text-foreground">Members</h2>
          <Badge variant="outline">{members.length}</Badge>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-muted-foreground text-sm">Loading members...</div>
        ) : (
          <div className="space-y-2">
            {members.map((member) => (
              <div
                key={member.id}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {member.profiles.avatar_url ? (
                    <img
                      src={member.profiles.avatar_url}
                      alt={member.profiles.full_name}
                      className="w-8 h-8 rounded-full"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-mosaic-coral to-mosaic-gold flex items-center justify-center text-white text-xs font-medium">
                      {member.profiles.full_name?.[0]?.toUpperCase() || 'U'}
                    </div>
                  )}
                  <div>
                    <div className="text-sm font-medium text-foreground">
                      {member.profiles.full_name || 'Unnamed'}
                    </div>
                    <div className="text-xs text-muted-foreground">{member.profiles.email}</div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {canManage && member.user_id !== user?.id ? (
                    <Select
                      value={member.role_id}
                      onValueChange={(value) => handleRoleChange(member.id, value)}
                    >
                      <SelectTrigger className="w-[140px] h-8 text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {roles.map((role) => (
                          <SelectItem key={role.id} value={role.id} className="text-xs">
                            {role.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  ) : (
                    <Badge variant="outline" className="text-xs">
                      {member.roles?.name || 'Unknown'}
                    </Badge>
                  )}
                  {canManage && member.user_id !== user?.id && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      onClick={() => handleRemoveMember(member.id)}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
