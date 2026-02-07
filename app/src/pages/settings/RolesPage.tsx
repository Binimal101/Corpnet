import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FolderTree,
  ChevronRight,
  ChevronDown,
  Plus,
  Pencil,
  Trash2,
  Check,
  X,
  Shield,
  Users,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { useAuth } from '@/contexts/AuthContext'
import { supabase } from '@/lib/supabase'
import type { Role } from '@/types/auth'

interface TreeNode extends Role {
  children: TreeNode[]
  memberCount: number
}

function buildTree(roles: Role[], memberCounts: Record<string, number>): TreeNode[] {
  const map = new Map<string, TreeNode>()
  const roots: TreeNode[] = []

  for (const role of roles) {
    map.set(role.id, { ...role, children: [], memberCount: memberCounts[role.id] || 0 })
  }

  for (const node of map.values()) {
    if (node.parent_role_id && map.has(node.parent_role_id)) {
      map.get(node.parent_role_id)!.children.push(node)
    } else {
      roots.push(node)
    }
  }

  return roots
}

function RoleTreeNode({
  node,
  depth,
  isRoot,
  canManage,
  onAddChild,
  onRename,
  onDelete,
}: {
  node: TreeNode
  depth: number
  isRoot: boolean
  canManage: boolean
  onAddChild: (parentId: string) => void
  onRename: (roleId: string, newName: string) => void
  onDelete: (roleId: string) => void
}) {
  const [expanded, setExpanded] = useState(true)
  const [editing, setEditing] = useState(false)
  const [editName, setEditName] = useState(node.name)
  const hasChildren = node.children.length > 0

  const handleRename = () => {
    if (editName.trim() && editName.trim() !== node.name) {
      onRename(node.id, editName.trim())
    }
    setEditing(false)
  }

  return (
    <div>
      <div
        className="group flex items-center gap-1 py-1.5 px-2 rounded-lg hover:bg-muted/50 transition-colors"
        style={{ paddingLeft: `${depth * 24 + 8}px` }}
      >
        {/* Expand/collapse toggle */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-5 h-5 flex items-center justify-center text-muted-foreground shrink-0"
        >
          {hasChildren ? (
            expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />
          ) : (
            <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
          )}
        </button>

        {/* Role icon */}
        <div className={`w-6 h-6 rounded flex items-center justify-center shrink-0 ${
          isRoot
            ? 'bg-mosaic-purple/20 text-mosaic-purple'
            : depth === 1
            ? 'bg-mosaic-teal/20 text-mosaic-teal'
            : 'bg-muted text-muted-foreground'
        }`}>
          {isRoot ? <Shield className="w-3.5 h-3.5" /> : <Users className="w-3.5 h-3.5" />}
        </div>

        {/* Name */}
        {editing ? (
          <div className="flex items-center gap-1 flex-1 ml-1">
            <Input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleRename()
                if (e.key === 'Escape') { setEditing(false); setEditName(node.name) }
              }}
              className="h-7 text-sm py-0"
              autoFocus
            />
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={handleRename}>
              <Check className="w-3.5 h-3.5 text-green-500" />
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => { setEditing(false); setEditName(node.name) }}>
              <X className="w-3.5 h-3.5" />
            </Button>
          </div>
        ) : (
          <>
            <span className="text-sm font-medium text-foreground ml-1 flex-1">{node.name}</span>
            {node.memberCount > 0 && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 ml-1">
                {node.memberCount} {node.memberCount === 1 ? 'member' : 'members'}
              </Badge>
            )}
            {isRoot && (
              <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-mosaic-purple text-mosaic-purple ml-1">
                Root
              </Badge>
            )}
          </>
        )}

        {/* Action buttons */}
        {canManage && !editing && (
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity ml-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => onAddChild(node.id)}
              title="Add child role"
            >
              <Plus className="w-3.5 h-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() => { setEditing(true); setEditName(node.name) }}
              title="Rename"
            >
              <Pencil className="w-3.5 h-3.5" />
            </Button>
            {!isRoot && (
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                onClick={() => onDelete(node.id)}
                title="Delete role"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Children */}
      <AnimatePresence>
        {expanded && hasChildren && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            {node.children.map((child) => (
              <RoleTreeNode
                key={child.id}
                node={child}
                depth={depth + 1}
                isRoot={false}
                canManage={canManage}
                onAddChild={onAddChild}
                onRename={onRename}
                onDelete={onDelete}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function RolesPage() {
  const { org } = useAuth()
  const [roles, setRoles] = useState<Role[]>([])
  const [memberCounts, setMemberCounts] = useState<Record<string, number>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [addingTo, setAddingTo] = useState<string | null>(null)
  const [newRoleName, setNewRoleName] = useState('')
  const [error, setError] = useState('')

  const canManage = org?.is_root ?? false

  const fetchRoles = useCallback(async () => {
    if (!org) return

    const [{ data: rolesData }, { data: membersData }] = await Promise.all([
      supabase
        .from('roles')
        .select('*')
        .eq('organization_id', org.org_id)
        .order('created_at', { ascending: true }),
      supabase
        .from('organization_members')
        .select('role_id')
        .eq('organization_id', org.org_id),
    ])

    setRoles((rolesData as Role[]) || [])

    // Count members per role
    const counts: Record<string, number> = {}
    for (const m of membersData || []) {
      counts[m.role_id] = (counts[m.role_id] || 0) + 1
    }
    setMemberCounts(counts)
    setIsLoading(false)
  }, [org])

  useEffect(() => {
    fetchRoles()
  }, [fetchRoles])

  const handleAddChild = (parentId: string) => {
    setAddingTo(parentId)
    setNewRoleName('')
    setError('')
  }

  const handleCreateRole = async () => {
    if (!org || !addingTo || !newRoleName.trim()) return
    setError('')

    const { error: insertError } = await supabase
      .from('roles')
      .insert({
        organization_id: org.org_id,
        name: newRoleName.trim(),
        parent_role_id: addingTo,
      })

    if (insertError) {
      setError(insertError.message)
      return
    }

    setAddingTo(null)
    setNewRoleName('')
    fetchRoles()
  }

  const handleRename = async (roleId: string, newName: string) => {
    if (!org) return

    const { error: updateError } = await supabase
      .from('roles')
      .update({ name: newName })
      .eq('id', roleId)

    if (updateError) {
      setError(updateError.message)
      return
    }

    fetchRoles()
  }

  const handleDelete = async (roleId: string) => {
    if (!org) return

    // Check if role has members
    if (memberCounts[roleId] > 0) {
      setError('Cannot delete a role that has members assigned. Reassign members first.')
      return
    }

    // Check if role has children
    const hasChildren = roles.some(r => r.parent_role_id === roleId)
    if (hasChildren) {
      setError('Cannot delete a role that has child roles. Delete child roles first.')
      return
    }

    const { error: deleteError } = await supabase
      .from('roles')
      .delete()
      .eq('id', roleId)

    if (deleteError) {
      setError(deleteError.message)
      return
    }

    setError('')
    fetchRoles()
  }

  const tree = buildTree(roles, memberCounts)

  return (
    <div className="space-y-6">
      <div className="glass rounded-xl border border-border/50 p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <FolderTree className="w-5 h-5 text-mosaic-purple" />
            <h2 className="text-lg font-semibold text-foreground">Role Hierarchy</h2>
            <Badge variant="outline">{roles.length} roles</Badge>
          </div>
        </div>

        <p className="text-sm text-muted-foreground mb-4">
          Define your organization's access control hierarchy. Roles higher in the tree have access to all data below them.
        </p>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="text-center py-8 text-muted-foreground text-sm">Loading roles...</div>
        ) : (
          <div className="rounded-lg border border-border/50 bg-muted/20 p-2">
            {tree.map((root) => (
              <RoleTreeNode
                key={root.id}
                node={root}
                depth={0}
                isRoot={true}
                canManage={canManage}
                onAddChild={handleAddChild}
                onRename={handleRename}
                onDelete={handleDelete}
              />
            ))}
          </div>
        )}

        {/* Inline add role form */}
        <AnimatePresence>
          {addingTo && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mt-4 p-4 rounded-lg border border-mosaic-purple/20 bg-mosaic-purple/5"
            >
              <div className="text-sm text-muted-foreground mb-2">
                Adding child role under <strong>{roles.find(r => r.id === addingTo)?.name}</strong>
              </div>
              <div className="flex gap-2">
                <Input
                  placeholder="Role name (e.g. Team Lead)"
                  value={newRoleName}
                  onChange={(e) => setNewRoleName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleCreateRole()
                    if (e.key === 'Escape') setAddingTo(null)
                  }}
                  autoFocus
                />
                <Button
                  onClick={handleCreateRole}
                  disabled={!newRoleName.trim()}
                  className="bg-gradient-to-r from-mosaic-purple to-mosaic-teal hover:opacity-90"
                >
                  Add
                </Button>
                <Button variant="outline" onClick={() => setAddingTo(null)}>
                  Cancel
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
