package lpk

import (
	"net/url"
	"regexp"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

var (
	urlQueryPattern       = regexp.MustCompile(`https?://[^\s"'<>]+`)
	internalPathPattern   = regexp.MustCompile(`/(?:user_home|lzcsys|lzcapp|ssd|var/lib/lzc-ai-agent)/[^\s"'<>]*`)
	internalFlagPattern   = regexp.MustCompile(`(?:--(?:quantization|speculative-config|kv-cache-dtype|gpu-memory-utilization|max-num-batched-tokens|enforce-eager|linear-backend|compilation-config|spec-type|spec-draft-[^\s;]+|model-draft|mmproj|load-mode)|-cc\.[^\s;]+|-(?:ctkd|ctvd|ctk|ctv|lm))\b[^\n;]*`)
	internalImagePattern  = regexp.MustCompile(`(?:local|registry\.lazycat\.cloud(?::\d+)?)/[^\s"'<>]+`)
)

// SanitizePublicText removes internal service names, runtime image tags,
// optimization flags, signed URL query parameters, and host filesystem paths
// from user-facing logs, error messages, and diagnostics.
func SanitizePublicText(message string) string {
	if strings.TrimSpace(message) == "" {
		return ""
	}
	out := urlQueryPattern.ReplaceAllStringFunc(message, func(raw string) string {
		parsed, err := url.Parse(raw)
		if err != nil {
			return "[链接已隐藏]"
		}
		parsed.User = nil
		parsed.RawQuery = ""
		parsed.Fragment = ""
		return parsed.String()
	})
	out = internalImagePattern.ReplaceAllString(out, "[runtime-image]")
	out = internalFlagPattern.ReplaceAllString(out, "[runtime-param]")
	out = internalPathPattern.ReplaceAllString(out, "[storage-path]")
	out = strings.ReplaceAll(out, config.DefaultServiceName, config.PublicServiceName)
	out = strings.ReplaceAll(out, config.DeviceModelNamespace, config.PublicServiceName)
	return strings.TrimSpace(out)
}
