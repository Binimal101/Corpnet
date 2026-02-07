import { Check, ChevronsUpDown, Building2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'
import { useAuth } from '@/contexts/AuthContext'

export function OrgSwitcher() {
  const { organizations, activeOrg, switchOrg } = useAuth()

  if (organizations.length <= 1) {
    return activeOrg ? (
      <div className="flex items-center gap-2 px-2">
        <Building2 className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm font-medium">{activeOrg.org_domain}</span>
        {activeOrg.is_producer && (
          <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-mosaic-purple text-mosaic-purple">
            Producer
          </Badge>
        )}
      </div>
    ) : null
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-2">
          <Building2 className="w-4 h-4" />
          <span className="text-sm">{activeOrg?.org_domain || 'Select org'}</span>
          {activeOrg?.is_producer && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-mosaic-purple text-mosaic-purple">
              Producer
            </Badge>
          )}
          <ChevronsUpDown className="w-3 h-3 text-muted-foreground" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-56">
        <DropdownMenuLabel>Organizations</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {organizations.map((org) => (
          <DropdownMenuItem
            key={org.org_id}
            onClick={() => switchOrg(org.org_id)}
            className="flex items-center justify-between"
          >
            <div className="flex items-center gap-2">
              <span>{org.org_domain}</span>
              {org.is_producer && (
                <Badge variant="outline" className="text-[10px] px-1.5 py-0 border-mosaic-purple text-mosaic-purple">
                  Producer
                </Badge>
              )}
            </div>
            {org.org_id === activeOrg?.org_id && (
              <Check className="w-4 h-4 text-mosaic-purple" />
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
