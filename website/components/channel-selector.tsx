"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

interface ChannelInfo {
  id: number
  slot: number
  unit: string
  model: string
}

interface ChannelSelectorProps {
  channels: ChannelInfo[]
  selectedChannel: number | null
  onChannelSelect: (channelId: number | null) => void
}

export function ChannelSelector({ channels, selectedChannel, onChannelSelect }: ChannelSelectorProps) {
  const currentChannels = channels.filter((ch) => ch.unit === "A")
  const otherChannels = channels.filter((ch) => ch.unit !== "A")

  if (currentChannels.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>No Current Channels Found</CardTitle>
          <CardDescription>
            No channels with current measurements were found in this DLog file.
          </CardDescription>
        </CardHeader>
        {otherChannels.length > 0 && (
          <CardContent>
            <p className="text-sm text-muted-foreground mb-2">Other channels found (voltage measurements):</p>
            <div className="space-y-2">
              {otherChannels.map((channel) => (
                <div key={`${channel.id}-${channel.unit}`} className="flex items-center justify-between p-2 bg-muted rounded">
                  <div>
                    <span className="font-medium">Channel {channel.id}</span>
                    <span className="text-muted-foreground ml-2">
                      Slot {channel.slot} • {channel.model}
                    </span>
                  </div>
                  <Badge variant="outline">Voltage</Badge>
                </div>
              ))}
            </div>
          </CardContent>
        )}
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Select Current Channel</CardTitle>
        <CardDescription>Choose one channel containing current measurements to convert (PPK2 supports only one channel)</CardDescription>
      </CardHeader>
      <CardContent>
        <RadioGroup 
          value={selectedChannel?.toString() || ""} 
          onValueChange={(value) => onChannelSelect(value ? parseInt(value) : null)}
        >
          <div className="space-y-3">
            {currentChannels.map((channel) => (
              <div key={channel.id} className="flex items-center space-x-3 p-3 rounded-lg border border-border hover:border-primary hover:bg-accent/50 transition-all duration-200">
                <RadioGroupItem value={channel.id.toString()} id={`channel-${channel.id}`} />
                <Label htmlFor={`channel-${channel.id}`} className="flex-1 cursor-pointer">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <span className="font-medium">Channel {channel.id}</span>
                      <span className="text-muted-foreground ml-2">
                        Slot {channel.slot} • {channel.model}
                      </span>
                    </div>
                    <div className="ml-4">
                      <Badge variant="secondary">Current</Badge>
                    </div>
                  </div>
                </Label>
              </div>
            ))}
          </div>
        </RadioGroup>

        {otherChannels.length > 0 && (
          <div className="mt-4 pt-4 border-t">
            <p className="text-sm text-muted-foreground mb-2">Other channels found (voltage measurements):</p>
            <div className="space-y-2">
              {otherChannels.map((channel) => (
                <div key={`${channel.id}-${channel.unit}`} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg border border-transparent hover:border-border hover:bg-muted transition-all duration-200">
                  <div className="flex-1">
                    <span className="font-medium">Channel {channel.id}</span>
                    <span className="text-muted-foreground ml-2">
                      Slot {channel.slot} • {channel.model}
                    </span>
                  </div>
                  <div className="ml-4">
                    <Badge variant="outline">Voltage</Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
