import { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { filterProfileSettings } from '@/utils/filter-profile-settings';
import { useCommonStore, type Profile } from '@/stores/common-store';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { X, Copy, RotateCcw } from 'lucide-react';
import { useParams, useSearch } from '@tanstack/react-router';
import { useDirectionsQuery } from '@/hooks/use-directions-queries';
import { useIsochronesQuery } from '@/hooks/use-isochrones-queries';
import { ServerSettings } from '@/components/settings-panel/server-settings';

type ProfileWithSettings = Exclude<Profile, 'auto'>;

export const SettingsPanel = () => {
  const { profile } = useSearch({ from: '/$activeTab' });
  const { activeTab } = useParams({ from: '/$activeTab' });
  const settings = useCommonStore((state) => state.settings);
  const settingsPanelOpen = useCommonStore((state) => state.settingsPanelOpen);
  const resetSettings = useCommonStore((state) => state.resetSettings);
  const toggleSettings = useCommonStore((state) => state.toggleSettings);
  const [copied, setCopied] = useState(false);
  const { refetch: refetchDirections } = useDirectionsQuery();
  const { refetch: refetchIsochrones } = useIsochronesQuery();

  const handleCopySettings = useCallback(async () => {
    const text = JSON.stringify(
      filterProfileSettings(profile as ProfileWithSettings, settings)
    );
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => {
      setCopied(false);
    }, 1000);
  }, [profile, settings]);

  const resetConfigSettings = useCallback(() => {
    resetSettings(profile || 'bicycle');
    if (activeTab === 'directions') {
      refetchDirections();
    } else {
      refetchIsochrones();
    }
  }, [activeTab, profile, resetSettings, refetchDirections, refetchIsochrones]);

  return (
    <Sheet open={settingsPanelOpen} modal={false}>
      <SheetContent
        side="right"
        className="w-[350px] pb-6 sm:max-w-[unset] max-h-screen overflow-y-scroll"
      >
        <SheetHeader className="justify-between">
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription className="sr-only">
            Settings for the current profile
          </SheetDescription>
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleSettings}
            data-testid="close-settings-button"
          >
            <X className="size-4" />
          </Button>
        </SheetHeader>
        <div className="px-3 space-y-3">
          <ServerSettings />
          <div className="flex gap-2 pt-1">
            <Button
              variant={copied ? 'default' : 'outline'}
              size="sm"
              onClick={handleCopySettings}
              className={copied ? 'bg-green-600 hover:bg-green-600' : ''}
            >
              <Copy className="size-3.5" />
              {copied ? 'Copied!' : 'Copy to Clipboard'}
            </Button>
            <Button variant="outline" size="sm" onClick={resetConfigSettings}>
              <RotateCcw className="size-3.5" />
              Reset
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};
