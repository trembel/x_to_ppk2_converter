/**
 * PPK2 File Converter - Core Processing Library
 * 
 * This module handles the conversion of CSV and DLog files to the Nordic PPK2 format.
 * It implements the exact same logic as the Python scripts for binary compatibility.
 * 
 * Key features:
 * - CSV parsing with configurable timestamp/current units
 * - DLog binary data extraction and channel parsing
 * - MiniFoldingBuffer implementation for PPK2 minimap generation
 * - Proper endianness handling (DLog big-endian → PPK2 little-endian)
 */

interface ChannelInfo {
  id: number
  slot: number
  unit: string
  model: string
  dataIndex?: number
}

interface MiniFoldingBuffer {
  maxElements: number
  numTimesToFold: number
  lastElementFoldCount: number
  data: {
    length: number
    min: Array<{ x: number; y: number } | null>
    max: Array<{ x: number; y: number } | null>
  }
}

export class FileProcessor {
  private PPK2_FORMAT_VERSION = 2
  private FRAME_SIZE_BYTES = 6
  private DEFAULT_DIGITAL_BITS = 0xaaaa

  /**
   * Analyzes a CSV file and returns the column headers
   */
  async analyzeCSVFile(file: File): Promise<string[]> {
    const text = await file.text()
    const lines = text.split("\n").filter(line => line.trim())

    if (lines.length === 0) {
      throw new Error("CSV file is empty")
    }

    const header = lines[0].split(",").map((col) => col.trim().replace(/"/g, ""))

    if (header.length === 0) {
      throw new Error("CSV file has no columns")
    }

    

    return header
  }

  /**
   * Analyzes a DLog file and extracts available channels
   */
  async analyzeDLogFile(file: File): Promise<ChannelInfo[]> {
    const arrayBuffer = await file.arrayBuffer()
    let data: Uint8Array

    // Handle .xz compression
    if (file.name.toLowerCase().endsWith(".xz")) {
      throw new Error("XZ compressed files are not yet supported in the browser version. Please decompress the file first.")
    } else {
      data = new Uint8Array(arrayBuffer)
    }

    // Find the end of XML header by searching for </dlog>
    const decoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: false })
    let xmlText = ""
    let foundEnd = false

    // Read data in chunks to find the XML end marker
    for (let i = 0; i < Math.min(data.length, 100000); i += 1000) {
      const chunk = decoder.decode(data.slice(0, i + 1000), { stream: false })
      if (chunk.includes("</dlog>")) {
        const endIndex = chunk.indexOf("</dlog>") + 7 // Include the closing tag
        xmlText = chunk.substring(0, endIndex)
        foundEnd = true
        break
      }
    }

    if (!foundEnd) {
      throw new Error("Could not find end of XML header (</dlog>) in DLog file")
    }

    // Parse XML with proper error handling
    const parser = new DOMParser()
    // Handle problematic numeric tags that aren't valid XML element names
    const cleanXml = xmlText.replace(/<1ua>/g, "<X1ua>").replace(/<\/1ua>/g, "</X1ua>").replace(/<2ua>/g, "<X2ua>").replace(/<\/2ua>/g, "</X2ua>")
    const xmlDoc = parser.parseFromString(cleanXml, "text/xml")

    const parseError = xmlDoc.querySelector("parsererror")
    if (parseError) {
      throw new Error(`Failed to parse DLog XML header: ${parseError.textContent}`)
    }

    const channels: ChannelInfo[] = []
    const channelElements = xmlDoc.querySelectorAll("channel")

    channelElements.forEach((channelEl) => {
      try {
        const id = Number.parseInt(channelEl.getAttribute("id") || "0")
        const modelEl = channelEl.querySelector("ident model")
        const slotEl = channelEl.querySelector("ident slot")
        const senseCurrEl = channelEl.querySelector("sense_curr")
        const senseVoltEl = channelEl.querySelector("sense_volt")

        const model = modelEl?.textContent?.trim() || "Unknown"
        const slotText = slotEl?.textContent?.trim() || "0"
        const slot = /^\d+$/.test(slotText) ? Number.parseInt(slotText) : 0
        const senseCurr = senseCurrEl?.textContent?.trim() === "1"
        const senseVolt = senseVoltEl?.textContent?.trim() === "1"

        // Add channels in the same order as Python: voltage first, then current
        if (senseVolt) {
          channels.push({ id, slot, unit: "V", model })
        }
        if (senseCurr) {
          channels.push({ id, slot, unit: "A", model })
        }
      } catch (error) {
        console.warn(`Error parsing channel with ID ${channelEl.getAttribute("id")}:`, error)
      }
    })

    if (channels.length === 0) {
      console.warn("No channels found or parsed successfully in DLog header")
    }

    return channels
  }

  /**
   * Converts a CSV file to PPK2 format
   * @param file - CSV file to convert
   * @param timestampColumn - Column name/index for timestamps
   * @param currentColumn - Column name/index for current values
   * @param currentUnit - Unit of current values (A/mA/µA/nA)
   * @param timeUnit - Unit of timestamp values (s/ms/µs)
   * @param startTimeStr - ISO timestamp for recording start time
   * @param onProgress - Progress callback function
   */
  async convertCSVToPPK2(
    file: File,
    timestampColumn: string,
    currentColumn: string,
    currentUnit: string,
    timeUnit: string,
    startTimeStr: string,
    onProgress: (progress: number, message: string) => void,
  ): Promise<Blob> {
    onProgress(10, "Reading CSV file...")

    const text = await file.text()
    const lines = text.split("\n").filter((line) => line.trim())

    if (lines.length < 2) {
      throw new Error("CSV file must have at least a header and one data row")
    }

    // Parse header and clean column names
    const header = lines[0].split(",").map((col) => col.trim().replace(/^"|"$/g, ""))
    console.log(`CSV Header: ${JSON.stringify(header)}`);

    // Find column indices using helper function
    const timestampIndex = this.findColumnIndex(header, timestampColumn)
    const currentIndex = this.findColumnIndex(header, currentColumn)

    if (timestampIndex === currentIndex) {
      throw new Error("Timestamp and current columns cannot be the same")
    }

    console.log(`Using timestamp column: '${header[timestampIndex]}' (index ${timestampIndex})`);
    console.log(`Using current column: '${header[currentIndex]}' (index ${currentIndex})`);

    onProgress(20, "Processing CSV data...")

    // 1. Validate Start Time and Convert to Milliseconds Epoch (exact Python logic)
    let startTimeMs: number
    try {
      // Assume ISO 8601 format, try parsing with and without timezone awareness
      let startDt: Date
      try {
        startDt = new Date(startTimeStr)
        if (isNaN(startDt.getTime())) {
          throw new Error("Invalid date")
        }
      } catch {
        // Try adding a 'Z' for UTC if timezone is missing (like Python)
        if (!startTimeStr.endsWith('Z') && !startTimeStr.includes('+') && !startTimeStr.slice(10).includes('-')) {
          startDt = new Date(startTimeStr + 'Z')
          if (isNaN(startDt.getTime())) {
            throw new Error("Invalid date with Z suffix")
          }
        } else {
          throw new Error("Cannot parse date")
        }
      }

      // If timezone naive, assume local timezone and convert to UTC (like Python)
      if (!startTimeStr.includes('Z') && !startTimeStr.includes('+') && !startTimeStr.slice(10).includes('-')) {
        console.warn("Start time has no timezone info, assuming local time.")
        // JavaScript Date constructor treats ISO strings without timezone as local time
        startDt = new Date(startTimeStr)
      }

      startTimeMs = startDt.getTime()
      console.log(`  Parsed Start Time (UTC): ${new Date(startTimeMs).toISOString()}`);
    } catch (error) {
      throw new Error(`Invalid start time format: ${startTimeStr}. Please use ISO 8601 format (e.g., '2023-10-27T10:00:00Z' or '2023-10-27T12:00:00+02:00'). Details: ${error}`);
    }

    // Process data
    const sessionData: number[] = []
    const minimap = this.createMiniFoldingBuffer()
    let samplesPerSecond = 0
    let firstTimestamp: number | null = null
    const sampleDeltas: number[] = []
    let lastTimestamp: number | null = null
    let rowCount = 1 // Start at 1 for header
    let skippedRows = 0

    console.log("Processing CSV rows...");

    for (let i = 1; i < lines.length; i++) {
      rowCount++
      
      if (i % 1000 === 0) {
        onProgress(20 + (i / lines.length) * 60, `Processing row ${i}/${lines.length}...`)
      }

      const row = lines[i].split(",")
      
      // Check if row has enough columns
      if (!row || row.length <= Math.max(timestampIndex, currentIndex)) {
        console.warn(`Skipping row ${rowCount} due to insufficient columns or empty row. Expected at least ${Math.max(timestampIndex, currentIndex) + 1} columns, got ${row.length}`);
        skippedRows++
        continue
      }

      try {
        // Parse timestamp and current - matching Python
        let timestamp = Number.parseFloat(row[timestampIndex].trim())
        let current = Number.parseFloat(row[currentIndex].trim())
        
        // Convert timestamp to seconds based on selected unit
        switch (timeUnit) {
          case "ms":
            timestamp = timestamp / 1000 // ms to s
            break
          case "µs":
            timestamp = timestamp / 1_000_000 // µs to s
            break
          case "s":
          default:
            // Already in seconds, no conversion needed
            break
        }
        
        // Convert current to amperes based on selected unit
        switch (currentUnit) {
          case "mA":
            current = current / 1000 // mA to A
            break
          case "µA":
            current = current / 1_000_000 // µA to A
            break
          case "nA":
            current = current / 1_000_000_000 // nA to A
            break
          case "A":
          default:
            // Already in amperes, no conversion needed
            break
        }

        if (!Number.isFinite(timestamp) || !Number.isFinite(current)) {
          skippedRows++
          continue
        }

        // Store first timestamp
        if (firstTimestamp === null) {
          firstTimestamp = timestamp
          console.log(`First data timestamp: ${firstTimestamp.toFixed(6)} s`);
        }

        // Calculate sample rate based on first ~100 samples (matching Python)
        if (lastTimestamp !== null && sampleDeltas.length < 100) {
          const delta = timestamp - lastTimestamp
          if (delta > 1e-9) { // Avoid division by zero or negative delta
            sampleDeltas.push(delta)
          } else if (delta <= 0) {
            console.warn(`Non-increasing timestamp detected at row ${rowCount}: ${timestamp} <= ${lastTimestamp}`);
          }
        }

        // Determine samples per second after enough deltas
        if (samplesPerSecond === 0 && sampleDeltas.length >= 10) {
          const avgDelta = sampleDeltas.reduce((a, b) => a + b, 0) / sampleDeltas.length
          
          // Check variance (optional validation)
          const variance = sampleDeltas.reduce((sum, delta) => sum + Math.pow(delta - avgDelta, 2), 0) / sampleDeltas.length
          const stdDev = Math.sqrt(variance)
          if (stdDev / avgDelta > 0.05) { // Allow 5% variation
            console.warn(`Sample rate appears inconsistent (avg delta: ${avgDelta.toFixed(6)}, std dev: ${stdDev.toFixed(6)}). Using average rate.`);
          }

          samplesPerSecond = Math.round(1.0 / avgDelta)
          console.log(`Determined Sample Rate: ${samplesPerSecond} Hz (average delta: ${(avgDelta * 1000).toFixed(5)} ms)`);
          // Clear deltas to stop checking rate
          sampleDeltas.length = 0
        }

        // Convert current to microamps for processing/storage
        const currentMicroAmps = current * 1_000_000

        // Add to minimap (relative timestamp from start)
        const relativeTime = timestamp - firstTimestamp
        this.addToMiniFoldingBuffer(minimap, currentMicroAmps, relativeTime)

        // Pack data for session.raw (float 32-bit little-endian current in microamps, unsigned 16-bit little-endian bits)
        // IMPORTANT: PPK2 uses Little Endian, must use DataView to ensure correct byte order
        const frameBuffer = new ArrayBuffer(6) // 4 bytes float + 2 bytes uint16
        const frameView = new DataView(frameBuffer)
        frameView.setFloat32(0, currentMicroAmps, true) // true = little-endian
        frameView.setUint16(4, this.DEFAULT_DIGITAL_BITS, true) // true = little-endian
        
        sessionData.push(...Array.from(new Uint8Array(frameBuffer)))

        lastTimestamp = timestamp

        if (rowCount % 50000 === 0) { // Print progress like Python
          console.log(`Processed ${rowCount} rows...`);
        }

      } catch (error) {
        console.warn(`Skipping row ${rowCount} due to parsing error: ${error}. Row data: ${JSON.stringify(row)}`);
        skippedRows++
        continue
      }
    }

    // Handle cases where sample rate couldn't be determined (matching Python logic)
    if (samplesPerSecond === 0) {
      if (sampleDeltas.length > 0) {
        // Handle cases with very few samples (< 10)
        const avgDelta = sampleDeltas.reduce((a, b) => a + b, 0) / sampleDeltas.length
        if (avgDelta <= 1e-9) {
          throw new Error(`Cannot determine sample rate. Average time delta is too small or non-positive (${avgDelta.toExponential(3)} s). Check timestamp column.`)
        }
        samplesPerSecond = Math.round(1.0 / avgDelta)
        console.log(`Determined Sample Rate (few samples): ${samplesPerSecond} Hz (average delta: ${(avgDelta * 1000).toFixed(5)} ms)`);
      } else if (rowCount <= 2) { // Header + 0 or 1 data row
        throw new Error("Cannot determine sample rate. Need at least 2 data rows in CSV.")
      } else {
        throw new Error("Could not determine sample rate after processing all rows.")
      }
    }

    const totalSamples = sessionData.length / this.FRAME_SIZE_BYTES
    console.log(`Finished processing ${rowCount} CSV rows. Total samples: ${totalSamples}`);
    
    if (skippedRows > 0) {
      console.log(`Skipped ${skippedRows} invalid/problematic rows.`);
    }

    if (sessionData.length === 0) {
      throw new Error("No valid data rows found or processed")
    }

    onProgress(85, "Creating PPK2 file...")

    return this.createPPK2File(sessionData, minimap, samplesPerSecond, startTimeMs, file.name, "CSV")
  }

  /**
   * Converts a DLog file to PPK2 format
   * @param file - DLog file to convert
   * @param channelId - ID of the current channel to extract
   * @param startTimeStr - ISO timestamp for recording start time
   * @param onProgress - Progress callback function
   */
  async convertDLogToPPK2(
    file: File,
    channelId: number,
    startTimeStr: string,
    onProgress: (progress: number, message: string) => void,
  ): Promise<Blob> {
    onProgress(10, "Reading DLog file...")

    const arrayBuffer = await file.arrayBuffer()
    const data = new Uint8Array(arrayBuffer)

    // Parse header to find XML end and extract metadata - match Python logic exactly
    const decoder = new TextDecoder("utf-8", { ignoreBOM: true, fatal: false })
    let xmlByteEnd = -1
    let foundEnd = false

    // Find XML header end by searching raw bytes (matching Python logic exactly)
    const dlogEndPattern = new Uint8Array([60, 47, 100, 108, 111, 103, 62]) // "</dlog>" in UTF-8 bytes
    
    // Search for </dlog> in the raw bytes
    for (let i = 0; i <= data.length - dlogEndPattern.length; i++) {
      let matches = true
      for (let j = 0; j < dlogEndPattern.length; j++) {
        if (data[i + j] !== dlogEndPattern[j]) {
          matches = false
          break
        }
      }
      if (matches) {
        xmlByteEnd = i + dlogEndPattern.length
        // Python reads line-by-line, so after </dlog> there might be newline characters
        // Skip any trailing newline characters (CR/LF) to match Python's f.read() behavior
        while (xmlByteEnd < data.length && (data[xmlByteEnd] === 10 || data[xmlByteEnd] === 13)) {
          xmlByteEnd++
        }
        foundEnd = true
        break
      }
    }

    if (!foundEnd || xmlByteEnd === -1) {
      throw new Error("Could not find end of XML header in DLog file")
    }

    // Extract XML text for parsing
    const xmlBytes = data.slice(0, xmlByteEnd)
    const xmlText = decoder.decode(xmlBytes)

    onProgress(20, "Parsing DLog header...")

    // Parse XML
    const parser = new DOMParser()
    const cleanXml = xmlText.replace(/<1ua>/g, "<X1ua>").replace(/<\/1ua>/g, "</X1ua>").replace(/<2ua>/g, "<X2ua>").replace(/<\/2ua>/g, "</X2ua>")
    const xmlDoc = parser.parseFromString(cleanXml, "text/xml")

    const parseError = xmlDoc.querySelector("parsererror")
    if (parseError) {
      throw new Error(`Failed to parse DLog XML: ${parseError.textContent}`)
    }

    // Extract sample interval and sense_minmax flag
    const frameEl = xmlDoc.querySelector("frame")
    if (!frameEl) {
      throw new Error("Could not find <frame> element in DLog header")
    }

    const tintEl = frameEl.querySelector("tint")
    const minmaxEl = frameEl.querySelector("sense_minmax")
    
    if (!tintEl?.textContent) {
      throw new Error("Could not find sample interval (tint) in DLog header")
    }

    const sampleInterval = Number.parseFloat(tintEl.textContent)
    if (sampleInterval <= 0) {
      throw new Error(`Invalid sample interval: ${sampleInterval}`)
    }
    const samplesPerSecond = Math.round(1.0 / sampleInterval)
    const senseMinMax = minmaxEl?.textContent?.trim() === "1"


    // Build channel list with data indices
    const parsedChannels: (ChannelInfo & { dataIndex: number })[] = []
    const channelElements = xmlDoc.querySelectorAll("channel")
    let dataIndexCounter = 0

    channelElements.forEach((channelEl) => {
      try {
        const id = Number.parseInt(channelEl.getAttribute("id") || "0")
        const modelEl = channelEl.querySelector("ident model")
        const slotEl = channelEl.querySelector("ident slot")
        const senseCurrEl = channelEl.querySelector("sense_curr")
        const senseVoltEl = channelEl.querySelector("sense_volt")

        const model = modelEl?.textContent?.trim() || "Unknown"
        const slotText = slotEl?.textContent?.trim() || "0"
        const slot = /^\d+$/.test(slotText) ? Number.parseInt(slotText) : 0
        const senseCurr = senseCurrEl?.textContent?.trim() === "1"
        const senseVolt = senseVoltEl?.textContent?.trim() === "1"

        // Add channels in order: voltage first, then current (matching Python)
        if (senseVolt) {
          parsedChannels.push({ id, slot, unit: "V", model, dataIndex: dataIndexCounter })
          dataIndexCounter++
        }
        if (senseCurr) {
          parsedChannels.push({ id, slot, unit: "A", model, dataIndex: dataIndexCounter })
          dataIndexCounter++
        }
      } catch (error) {
        console.warn(`Error parsing channel ${channelEl.getAttribute("id")}:`, error)
      }
    })

    if (parsedChannels.length === 0) {
      throw new Error("No valid channels found in DLog header")
    }

    // Find target current channel
    const targetChannel = parsedChannels.find((ch) => ch.id === channelId && ch.unit === "A")
    if (!targetChannel) {
      const availableCurrentChannels = parsedChannels.filter(ch => ch.unit === "A")
      const channelList = availableCurrentChannels.map(ch => `ID=${ch.id} Slot=${ch.slot} Model='${ch.model}'`).join(", ")
      throw new Error(`Could not find current channel with ID ${channelId}. Available current channels: ${channelList || "None"}`)
    }


    onProgress(40, "Processing binary data...")

    // Calculate data stream parameters
    const numParsedChannels = parsedChannels.length
    const numDataStreams = senseMinMax ? numParsedChannels * 3 : numParsedChannels // avg, min, max if senseMinMax
    const bytesPerSample = 4 * numDataStreams // 4 bytes per float32
    
    // Extract binary data starting after XML
    const binaryData = data.slice(xmlByteEnd)
    const numSamples = Math.floor(binaryData.length / bytesPerSample)


    if (numSamples === 0) {
      throw new Error(`No binary data found. Expected ${bytesPerSample} bytes per sample, but only ${binaryData.length} bytes available.`)
    }

    if (binaryData.length % bytesPerSample !== 0) {
      const remainder = binaryData.length % bytesPerSample
      console.warn(`Binary data size (${binaryData.length}) is not exact multiple of bytes per sample (${bytesPerSample}). Remainder: ${remainder} bytes. Using ${numSamples} complete samples.`)
      // Truncate to complete samples like Python does
      const truncatedLength = numSamples * bytesPerSample
    }

    // Determine target data index in the flattened array
    let targetDataIndex = targetChannel.dataIndex
    if (senseMinMax) {
      targetDataIndex *= 3 // Use avg stream (index * 3 + 0)
    }

    if (targetDataIndex >= numDataStreams) {
      throw new Error(`Target data index ${targetDataIndex} is out of bounds for ${numDataStreams} data streams`)
    }

    // Extract all data first, then reshape like Python (matching numpy logic)
    // Python: np_data_flat = np.frombuffer(raw_data, dtype='>f')
    // Python: np_data_reshaped = np_data_flat.reshape((num_samples, num_data_streams)).T
    
    // Truncate binary data to complete samples like Python does
    const completeDataLength = numSamples * bytesPerSample
    const truncatedBinaryData = binaryData.slice(0, completeDataLength)
    
    // Read all float32 values as big-endian (matching Python '>f')
    const allFloats = new Float32Array(numSamples * numDataStreams)
    for (let i = 0; i < numSamples * numDataStreams; i++) {
      const byteOffset = i * 4
      const dataView = new DataView(truncatedBinaryData.buffer, truncatedBinaryData.byteOffset + byteOffset, 4)
      allFloats[i] = dataView.getFloat32(0, false) // false = big-endian
    }
    
    // Reshape and transpose like Python: (numSamples, numDataStreams) then .T
    // After transpose: allFloats[stream * numSamples + sample]
    // Extract target channel data (matching Python: dlog_data_np[target_idx])
    const targetCurrentDataAmps = new Float32Array(numSamples)
    for (let i = 0; i < numSamples; i++) {
      // Python access after transpose: dlog_data_np[target_idx][i]
      // Equivalent: allFloats[target_idx * numSamples + i]
      targetCurrentDataAmps[i] = allFloats[targetDataIndex * numSamples + i]
    }
    
    // Calculate range manually to avoid 'too many arguments' error with large arrays
    let minValue = Number.POSITIVE_INFINITY
    let maxValue = Number.NEGATIVE_INFINITY
    for (let i = 0; i < targetCurrentDataAmps.length; i++) {
      const val = targetCurrentDataAmps[i]
      if (val < minValue) minValue = val
      if (val > maxValue) maxValue = val
    }
    
    // Process extracted channel data (matching Python loop)
    const sessionData: number[] = []
    const minimap = this.createMiniFoldingBuffer()
    let skippedSamples = 0

    for (let i = 0; i < numSamples; i++) {
      if (i % 10000 === 0) {
        onProgress(40 + (i / numSamples) * 40, `Processing sample ${i + 1}/${numSamples}...`)
      }

      // Get current value from extracted channel data (matching Python)
      const currentAmps = targetCurrentDataAmps[i]

      // Check for invalid values (matching Python logic)
      if (!Number.isFinite(currentAmps) || Math.abs(currentAmps) > 1e7) {
        if (skippedSamples < 10) {
          console.warn(`Skipping invalid data value ${currentAmps} at sample ${i}, replacing with 0`)
        } else if (skippedSamples === 10) {
          console.warn("Suppressing further invalid data value warnings...")
        }
        // Replace invalid values with 0 (matching Python behavior)
        const currentMicroAmps = 0
        const relativeTime = i * sampleInterval
        this.addToMiniFoldingBuffer(minimap, currentMicroAmps, relativeTime)
        
        // Pack zero value
        const currentBytes = new Float32Array([currentMicroAmps])
        const currentBuffer = new Uint8Array(currentBytes.buffer)
        const bitsBuffer = new Uint8Array(2)
        new DataView(bitsBuffer.buffer).setUint16(0, this.DEFAULT_DIGITAL_BITS, true)
        sessionData.push(...Array.from(currentBuffer), ...Array.from(bitsBuffer))
        
        skippedSamples++
        continue
      }

      const currentMicroAmps = currentAmps * 1_000_000
      const relativeTime = i * sampleInterval

      // Add to minimap
      this.addToMiniFoldingBuffer(minimap, currentMicroAmps, relativeTime)

      // Pack data for session.raw (little-endian for PPK2)
      // IMPORTANT: PPK2 uses Little Endian, must use DataView to ensure correct byte order
      const frameBuffer = new ArrayBuffer(6) // 4 bytes float + 2 bytes uint16
      const frameView = new DataView(frameBuffer)
      frameView.setFloat32(0, currentMicroAmps, true) // true = little-endian
      frameView.setUint16(4, this.DEFAULT_DIGITAL_BITS, true) // true = little-endian
      
      sessionData.push(...Array.from(new Uint8Array(frameBuffer)))
    }

    const totalSamplesWritten = sessionData.length / this.FRAME_SIZE_BYTES

    if (sessionData.length === 0) {
      throw new Error("No valid samples were processed")
    }

    onProgress(85, "Creating PPK2 file...")

    
    // 1. Validate Start Time and Convert to Milliseconds Epoch (exact Python logic)
    let startTimeMs: number
    try {
      // Assume ISO 8601 format, try parsing with and without timezone awareness
      let startDt: Date
      try {
        startDt = new Date(startTimeStr)
        if (isNaN(startDt.getTime())) {
          throw new Error("Invalid date")
        }
      } catch {
        // Try adding a 'Z' for UTC if timezone is missing (like Python)
        if (!startTimeStr.endsWith('Z') && !startTimeStr.includes('+') && !startTimeStr.slice(10).includes('-')) {
          startDt = new Date(startTimeStr + 'Z')
          if (isNaN(startDt.getTime())) {
            throw new Error("Invalid date with Z suffix")
          }
        } else {
          throw new Error("Cannot parse date")
        }
      }

      // If timezone naive, assume local timezone and convert to UTC (like Python)
      if (!startTimeStr.includes('Z') && !startTimeStr.includes('+') && !startTimeStr.slice(10).includes('-')) {
        console.warn("Start time has no timezone info, assuming local time.")
        // JavaScript Date constructor treats ISO strings without timezone as local time
        startDt = new Date(startTimeStr)
      }

      startTimeMs = startDt.getTime()
      console.log(`  Parsed Start Time (UTC): ${new Date(startTimeMs).toISOString()}`);
    } catch (error) {
      throw new Error(`Invalid start time format: ${startTimeStr}. Please use ISO 8601 format (e.g., '2023-10-27T10:00:00Z' or '2023-10-27T12:00:00+02:00'). Details: ${error}`);
    }
    console.log(`Using current system time: ${new Date(startTimeMs).toISOString()}`);

    return this.createPPK2File(sessionData, minimap, samplesPerSecond, startTimeMs, file.name, "Keysight DLog", {
      targetChannelId: channelId,
      targetChannelSlot: targetChannel.slot,
      targetChannelModel: targetChannel.model,
    })
  }

  private findColumnIndex(header: string[], columnSpec: string): number {
    // Try interpreting as 0-based index first (matching Python logic)
    const index = Number.parseInt(columnSpec)
    if (!isNaN(index)) {
      if (index >= 0 && index < header.length) {
        console.log(`Using 0-based index ${index} for column specification '${columnSpec}'.`);
        return index
      } else {
        throw new Error(`Column index '${index}' is out of range (0-${header.length - 1}).`)
      }
    }

    // Try as column name (case-insensitive matching)
    const lowerSpec = columnSpec.toLowerCase().trim()
    for (let i = 0; i < header.length; i++) {
      if (header[i].toLowerCase().trim() === lowerSpec) {
        console.log(`Found column by name: '${header[i]}' (index ${i}).`);
        return i
      }
    }

    throw new Error(`Could not find column with name '${columnSpec}'. Available columns: ${JSON.stringify(header)}`)
  }

  private createMiniFoldingBuffer(): MiniFoldingBuffer {
    return {
      maxElements: 10000,
      numTimesToFold: 1,
      lastElementFoldCount: 0,
      data: {
        length: 0,
        min: new Array(10000).fill(null),
        max: new Array(10000).fill(null),
      },
    }
  }

  private addToMiniFoldingBuffer(buffer: MiniFoldingBuffer, valueMicroAmps: number, timestampSec: number) {
    const timestampUs = timestampSec * 1_000_000
    const valueNa = valueMicroAmps * 1_000 // Convert to nanoamps

    // Apply floor value
    const clampedValue = Math.max(valueNa, 200)

    if (buffer.lastElementFoldCount === 0) {
      if (buffer.data.length === buffer.maxElements) {
        this.foldBuffer(buffer)
      }
      this.addDefaultToBuffer(buffer, timestampUs)
    }

    buffer.lastElementFoldCount++
    const currentIdx = buffer.data.length - 1

    // Update min point
    const minPoint = buffer.data.min[currentIdx]!
    const alpha = 1.0 / buffer.lastElementFoldCount
    minPoint.x = timestampUs * alpha + minPoint.x * (1 - alpha)
    minPoint.y = Math.min(clampedValue, minPoint.y)

    // Update max point
    const maxPoint = buffer.data.max[currentIdx]!
    maxPoint.x = timestampUs * alpha + maxPoint.x * (1 - alpha)
    maxPoint.y = Math.max(clampedValue, maxPoint.y)

    if (buffer.lastElementFoldCount === buffer.numTimesToFold) {
      buffer.lastElementFoldCount = 0
    }
  }

  private addDefaultToBuffer(buffer: MiniFoldingBuffer, timestampUs: number) {
    const idx = buffer.data.length
    if (idx < buffer.maxElements) {
      buffer.data.min[idx] = { x: timestampUs, y: Number.POSITIVE_INFINITY }
      buffer.data.max[idx] = { x: timestampUs, y: Number.NEGATIVE_INFINITY }
      buffer.data.length++
    }
  }

  private foldBuffer(buffer: MiniFoldingBuffer) {
    const newLength = Math.floor(buffer.data.length / 2)

    for (let i = 0; i < newLength; i++) {
      const idx1 = i * 2
      const idx2 = i * 2 + 1

      const avgX = (buffer.data.min[idx1]!.x + buffer.data.min[idx2]!.x) / 2
      const minY = Math.min(buffer.data.min[idx1]!.y, buffer.data.min[idx2]!.y)
      const maxY = Math.max(buffer.data.max[idx1]!.y, buffer.data.max[idx2]!.y)

      buffer.data.min[i] = { x: avgX, y: minY }
      buffer.data.max[i] = { x: avgX, y: maxY }
    }

    // Clear the rest
    for (let i = newLength; i < buffer.data.length; i++) {
      buffer.data.min[i] = null
      buffer.data.max[i] = null
    }

    buffer.data.length = newLength
    buffer.numTimesToFold *= 2
  }

  private async createPPK2File(
    sessionData: number[],
    minimap: MiniFoldingBuffer,
    samplesPerSecond: number,
    startTimeMs: number,
    filename: string,
    fileType: string,
    additionalInfo?: any,
  ): Promise<Blob> {
    // Create metadata (matching Python structure exactly)
    const metadata = {
      metadata: {
        samplesPerSecond,
        startSystemTime: startTimeMs,
        createdBy: "PPK2 Web Converter",
        sourceFileInfo: {
          filename: filename.split('/').pop() || filename, // Extract basename like Python
          type: fileType,
          ...additionalInfo,
        },
      },
      formatVersion: this.PPK2_FORMAT_VERSION,
    }

    // Create minimap data (matching Python MiniFoldingBuffer.get_state())
    const validMin = minimap.data.min.slice(0, minimap.data.length).filter((x) => x !== null)
    const validMax = minimap.data.max.slice(0, minimap.data.length).filter((x) => x !== null)
    
    // Ensure lengths match (matching Python validation)
    if (validMin.length !== minimap.data.length || validMax.length !== minimap.data.length) {
      console.warn(`Mismatch in valid min/max lengths (${validMin.length}, ${validMax.length}) vs data.length (${minimap.data.length})`);
    }

    const minimapData = {
      lastElementFoldCount: minimap.lastElementFoldCount,
      data: {
        length: minimap.data.length,
        min: validMin,
        max: validMax,
      },
      maxNumberOfElements: minimap.maxElements,
      numberOfTimesToFold: minimap.numTimesToFold,
    }


    // Create ZIP file using JSZip (matching Python's zipfile with DEFLATE compression level 6)
    const JSZip = (await import("jszip")).default
    const zip = new JSZip()

    // Use compact JSON encoding (no indentation, matching Python)
    zip.file("metadata.json", JSON.stringify(metadata, null, 0))
    zip.file("session.raw", new Uint8Array(sessionData))
    zip.file("minimap.raw", JSON.stringify(minimapData, null, 0))

    return await zip.generateAsync({ 
      type: "blob", 
      compression: "DEFLATE", 
      compressionOptions: { level: 6 } // Match Python compresslevel=6
    })
  }
}
