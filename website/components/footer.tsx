import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"

export function Footer() {
  return (
    <Card className="mt-8">
      <CardContent className="pt-4">
        <Separator className="mb-4" />
        <div className="text-sm text-muted-foreground space-y-2">
          <div className="flex flex-wrap items-center gap-4 justify-between">
            <div>
              <span className="font-medium">PPK2 Web Converter</span>
              <span className="mx-2">•</span>
              <span>Version: 141e6ed</span>
              <span className="mx-2">•</span>
              <span>© 2025 Silvano Cortesi</span>
            </div>
            <div className="flex items-center gap-4">
              <a 
                href="https://github.com/trembel/x_to_ppk2_converter/blob/main/LICENSE" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline"
              >
                LICENSE
              </a>
            </div>
          </div>
          <div className="text-xs">
            <p>
              <strong>Disclaimer:</strong> This software is provided "as is" without warranty of any kind. 
              Use at your own risk. Always verify converted files before using them in critical applications.
            </p>
            <p>
              <i>This web-app is completely AI-generated from the existing Python scripts</i>
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
