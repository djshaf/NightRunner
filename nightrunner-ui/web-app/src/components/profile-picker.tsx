import { ReactComponent as PedestrianSvg } from '@/images/pedestrian.svg';
import type { Profile } from '@/stores/common-store';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './ui/tooltip';
import { ToggleGroup, ToggleGroupItem } from './ui/toggle-group';
import { useCallback } from 'react';
import { useCommonStore } from '@/stores/common-store';
import { Loader2 } from 'lucide-react';
import { useSearch } from '@tanstack/react-router';

const iconMap = {
  pedestrian: <PedestrianSvg className="size-7" />,
};

interface ProfilePickerProps {
  loading: boolean;
  onProfileChange: (value: Profile) => void;
}

export const ProfilePicker = ({
  loading,
  onProfileChange,
}: ProfilePickerProps) => {
  const resetSettings = useCommonStore((state) => state.resetSettings);
  const { profile: activeProfile } = useSearch({ from: '/$activeTab' });

  const handleUpdateProfile = useCallback(
    (value: Profile) => {
      resetSettings(value);
      onProfileChange(value);
    },
    [resetSettings, onProfileChange]
  );

  const profiles = [{ value: 'pedestrian', label: 'Pedestrian' }];

  return (
    <div className="flex flex-col gap-2">
      <TooltipProvider>
        <ToggleGroup
          type="single"
          variant="outline"
          size="lg"
          value={activeProfile}
          className="[&_button]:h-12 [&_button]:min-w-12 [&_button]:px-1"
          onValueChange={(value: Profile) => {
            if (value && value !== activeProfile) {
              handleUpdateProfile(value);
            }
          }}
        >
          {profiles.map((profile, i) => (
            <Tooltip key={i}>
              <TooltipTrigger asChild>
                <ToggleGroupItem
                  value={profile.value}
                  aria-label={`Select ${profile.label} profile`}
                  data-testid={`profile-button-` + profile.value}
                  data-state={profile.value === activeProfile ? 'on' : 'off'}
                >
                  {profile.value === activeProfile && loading ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    iconMap[profile.value as keyof typeof iconMap]
                  )}
                </ToggleGroupItem>
              </TooltipTrigger>
              <TooltipContent>{profile.label}</TooltipContent>
            </Tooltip>
          ))}
        </ToggleGroup>
      </TooltipProvider>
    </div>
  );
};
