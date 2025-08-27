/**
 * PPK2 Converter - Main React Component
 * 
 * A web-based converter for CSV and DLog files to Nordic PPK2 format.
 * Provides drag-and-drop file upload, automatic channel detection,
 * configurable units, and progress tracking.
 * 
 * Features:
 * - CSV files: configurable timestamp/current units (s/ms/µs, A/mA/µA/nA)
 * - DLog files: automatic channel discovery with radio button selection  
 * - Real-time conversion progress with detailed status messages
 * - Binary-compatible output with Python reference implementation
 */
"use client"

import { useState, useCallback } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { FileUpload } from "@/components/file-upload"
import { ChannelSelector } from "@/components/channel-selector"
import { ConversionProgress } from "@/components/conversion-progress"
import { Footer } from "@/components/footer"
import { FileProcessor } from "@/lib/file-processor"
import { FileIcon, Download, AlertCircle, CheckCircle2, X, Github } from "lucide-react"

interface FileInfo {
  name: string
  size: number
  type: "csv" | "dlog"
  file: File
}

interface ChannelInfo {
  id: number
  slot: number
  unit: string
  model: string
}

interface ConversionState {
  status: "idle" | "analyzing" | "ready" | "converting" | "complete" | "error"
  progress: number
  message: string
  error?: string
  result?: Blob
}

export function PPK2Converter() {
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null)
  const [channels, setChannels] = useState<ChannelInfo[]>([])
  const [selectedChannel, setSelectedChannel] = useState<number | null>(null)
  const [csvColumns, setCsvColumns] = useState<string[]>([])
  const [timestampColumn, setTimestampColumn] = useState<string>("")
  const [currentColumn, setCurrentColumn] = useState<string>("") 
  const [currentUnit, setCurrentUnit] = useState<string>("A") // Default to Amperes
  const [timeUnit, setTimeUnit] = useState<string>("s") // Default to seconds
  const [startTime, setStartTime] = useState<string>("") // Empty by default, like Python --start-time
  const [conversion, setConversion] = useState<ConversionState>({
    status: "idle",
    progress: 0,
    message: "",
  })

  const handleFileSelect = useCallback(async (file: File) => {
    const fileType =
      file.name.toLowerCase().endsWith(".dlog") || file.name.toLowerCase().endsWith(".dlog.xz") ? "dlog" : "csv"

    setFileInfo({
      name: file.name,
      size: file.size,
      type: fileType,
      file,
    })

    setConversion({
      status: "analyzing",
      progress: 10,
      message: `Analyzing ${fileType.toUpperCase()} file...`,
    })

    try {
      const processor = new FileProcessor()

      if (fileType === "dlog") {
        const channels = await processor.analyzeDLogFile(file)
        setChannels(channels)
        
        // Filter for current channels only for the message
        const currentChannels = channels.filter(ch => ch.unit === "A")
        
        setConversion({
          status: "ready",
          progress: 100,
          message: channels.length > 0 
            ? `Found ${channels.length} measurement streams (${currentChannels.length} current channels). Select a current channel to continue.`
            : "No channels found in DLog file. Please check if the file is valid.",
        })
      } else {
        const columns = await processor.analyzeCSVFile(file)
        setCsvColumns(columns)
        
        setConversion({
          status: "ready",
          progress: 100,
          message: `Found ${columns.length} columns. Configure timestamp and current columns to continue.`,
        })
      }
    } catch (error) {
      setConversion({
        status: "error",
        progress: 0,
        message: "Failed to analyze file",
        error: error instanceof Error ? error.message : "Unknown error",
      })
    }
  }, [])

  const handleConvert = useCallback(async () => {
    if (!fileInfo) return

    setConversion({
      status: "converting",
      progress: 0,
      message: "Starting conversion...",
    })

    try {
      const processor = new FileProcessor()

      const onProgress = (progress: number, message: string) => {
        setConversion((prev) => ({
          ...prev,
          progress,
          message,
        }))
      }

      let result: Blob

      if (fileInfo.type === "dlog") {
        if (selectedChannel === null) {
          throw new Error("Please select a current channel")
        }
        // Use current time if empty (exactly like Python: datetime.now().isoformat())
        const actualStartTime = startTime.trim() || new Date().toISOString()
        result = await processor.convertDLogToPPK2(fileInfo.file, selectedChannel, actualStartTime, onProgress)
      } else {
        if (!timestampColumn || !currentColumn) {
          throw new Error("Please select timestamp and current columns")
        }
        // Use current time if empty (exactly like Python: datetime.now().isoformat())
        const actualStartTime = startTime.trim() || new Date().toISOString()
        result = await processor.convertCSVToPPK2(fileInfo.file, timestampColumn, currentColumn, currentUnit, timeUnit, actualStartTime, onProgress)
      }

      setConversion({
        status: "complete",
        progress: 100,
        message: "Conversion completed successfully!",
        result,
      })
    } catch (error) {
      setConversion({
        status: "error",
        progress: 0,
        message: "Conversion failed",
        error: error instanceof Error ? error.message : "Unknown error",
      })
    }
  }, [fileInfo, selectedChannel, timestampColumn, currentColumn, currentUnit, timeUnit, startTime])

  const handleDownload = useCallback(() => {
    if (!conversion.result || !fileInfo) return

    const url = URL.createObjectURL(conversion.result)
    const a = document.createElement("a")
    a.href = url
    a.download = fileInfo.name.replace(/\.(csv|dlog|dlog\.xz)$/i, ".ppk2")
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [conversion.result, fileInfo])

  const handleClearFile = useCallback(() => {
    setFileInfo(null)
    setChannels([])
    setCsvColumns([])
    setSelectedChannel(null)
    setTimestampColumn("")
    setCurrentColumn("")
    setCurrentUnit("A")
    setTimeUnit("s")
    setStartTime("")
    setConversion({
      status: "idle",
      progress: 0,
      message: "",
    })
  }, [])

  const canConvert = () => {
    if (!fileInfo || conversion.status !== "ready") return false

    if (fileInfo.type === "dlog") {
      return selectedChannel !== null
    } else {
      return timestampColumn && currentColumn
    }
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="text-center mb-8">
        <div className="flex items-center justify-center gap-4 mb-4">
          <h1 className="text-4xl font-bold text-foreground">PPK2 File Converter</h1>
          <a 
            href="https://github.com/trembel/x_to_ppk2_converter" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 transition-colors"
          >
            <Github className="h-4 w-4" />
            <span className="text-sm font-medium">GitHub</span>
          </a>
        </div>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Convert CSV and DLog measurement files to Nordic Power Profiler Kit 2 (PPK2) format. Simply drag and drop your
          file to get started.
        </p>
      </div>

      <div className="space-y-6">
        {/* File Upload */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileIcon className="h-5 w-5" />
              File Upload
            </CardTitle>
            <CardDescription>
              Upload a CSV file with timestamp and current columns, or a DLog file from Keysight instruments
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FileUpload onFileSelect={handleFileSelect} />

            {fileInfo && (
              <div className="mt-4 p-4 bg-muted rounded-lg">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">{fileInfo.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {(fileInfo.size / 1024 / 1024).toFixed(2)} MB • {fileInfo.type.toUpperCase()}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">{fileInfo.type.toUpperCase()}</Badge>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleClearFile}
                      className="h-8 w-8 p-0 hover:bg-destructive hover:text-destructive-foreground"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Analysis Progress */}
        {conversion.status === "analyzing" && (
          <ConversionProgress progress={conversion.progress} message={conversion.message} status={conversion.status} />
        )}

        {/* Configuration */}
        {conversion.status === "ready" && fileInfo && (
          <Card>
            <CardHeader>
              <CardTitle>Configuration</CardTitle>
              <CardDescription>
                Configure the conversion parameters for your {fileInfo.type.toUpperCase()} file
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="start-time">Start Time (Optional)</Label>
                <Input
                  id="start-time"
                  type="datetime-local"
                  value={startTime}
                  onChange={(e) => setStartTime(e.target.value)}
                  className="mt-1"
                  placeholder="Leave empty to use current system time"
                />
                <p className="text-sm text-muted-foreground mt-1">
                  Leave empty to use current system time (like Python script)
                </p>
              </div>

              {fileInfo.type === "dlog" ? (
                <ChannelSelector
                  channels={channels}
                  selectedChannel={selectedChannel}
                  onChannelSelect={setSelectedChannel}
                />
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="timestamp-column">Timestamp Column</Label>
                      <Select value={timestampColumn} onValueChange={setTimestampColumn}>
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select timestamp column" />
                        </SelectTrigger>
                        <SelectContent>
                          {csvColumns.map((column, index) => (
                            <SelectItem key={index} value={column}>
                              {column} (Index {index})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-muted-foreground mt-1">Column containing timestamps</p>
                    </div>

                    <div>
                      <Label htmlFor="current-column">Current Column</Label>
                      <Select value={currentColumn} onValueChange={setCurrentColumn}>
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select current column" />
                        </SelectTrigger>
                        <SelectContent>
                          {csvColumns.map((column, index) => (
                            <SelectItem key={index} value={column}>
                              {column} (Index {index})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-muted-foreground mt-1">Column containing current values</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="time-unit">Time Unit</Label>
                      <Select value={timeUnit} onValueChange={setTimeUnit}>
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select time unit" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="s">Seconds (s)</SelectItem>
                          <SelectItem value="ms">Milliseconds (ms)</SelectItem>
                          <SelectItem value="µs">Microseconds (µs)</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-muted-foreground mt-1">Unit of the timestamp column values</p>
                    </div>

                    <div>
                      <Label htmlFor="current-unit">Current Unit</Label>
                      <Select value={currentUnit} onValueChange={setCurrentUnit}>
                        <SelectTrigger className="mt-1">
                          <SelectValue placeholder="Select current unit" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="A">Amperes (A)</SelectItem>
                          <SelectItem value="mA">Milliamperes (mA)</SelectItem>
                          <SelectItem value="µA">Microamperes (µA)</SelectItem>
                          <SelectItem value="nA">Nanoamperes (nA)</SelectItem>
                        </SelectContent>
                      </Select>
                      <p className="text-sm text-muted-foreground mt-1">Unit of the current column values</p>
                    </div>
                  </div>
                </div>
              )}

              <Button onClick={handleConvert} disabled={!canConvert()} className="w-full" size="lg">
                Convert to PPK2
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Conversion Progress */}
        {conversion.status === "converting" && (
          <ConversionProgress progress={conversion.progress} message={conversion.message} status={conversion.status} />
        )}

        {/* Results */}
        {conversion.status === "complete" && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-primary">
                <CheckCircle2 className="h-5 w-5" />
                Conversion Complete
              </CardTitle>
              <CardDescription>Your file has been successfully converted to PPK2 format</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={handleDownload} size="lg" className="w-full">
                <Download className="h-4 w-4 mr-2" />
                Download PPK2 File
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Error */}
        {conversion.status === "error" && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              <strong>{conversion.message}</strong>
              {conversion.error && <div className="mt-2 text-sm">{conversion.error}</div>}
            </AlertDescription>
          </Alert>
        )}
      </div>

      <Footer />
    </div>
  )
}
