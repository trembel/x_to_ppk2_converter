"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Loader2, FileText } from "lucide-react"

interface ConversionProgressProps {
  progress: number
  message: string
  status: "analyzing" | "converting"
}

export function ConversionProgress({ progress, message, status }: ConversionProgressProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          {status === "analyzing" ? <FileText className="h-5 w-5" /> : <Loader2 className="h-5 w-5 animate-spin" />}
          {status === "analyzing" ? "Analyzing File" : "Converting to PPK2"}
        </CardTitle>
        <CardDescription>{message}</CardDescription>
      </CardHeader>
      <CardContent>
        <Progress value={progress} className="w-full" />
        <p className="text-sm text-muted-foreground mt-2 text-center">{progress}% complete</p>
      </CardContent>
    </Card>
  )
}
