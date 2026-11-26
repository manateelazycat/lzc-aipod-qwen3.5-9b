package lpk

import (
	"fmt"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

const (
	// Minimum physical RAM threshold for AGX Thor (64GB T4000 / 128GB T5000) and AGX Orin 64GB.
	MinSupportedTotalMemoryBytes = 48 << 30
)

// IsSupportedHardwareModel checks whether a Host device model/platform string belongs
// to the verified Jetson AGX Thor (Dev/T5000/T4000) or Jetson AGX Orin 64GB hardware family.
func IsSupportedHardwareModel(model, platform string) bool {
	m := strings.ToLower(strings.TrimSpace(model + " " + platform))
	return isThorModel(m) || isAGXOrinModel(m)
}

func isThorModel(m string) bool {
	return strings.Contains(m, "thor") || strings.Contains(m, "t5000") || strings.Contains(m, "t4000")
}

func isAGXOrinModel(m string) bool {
	if strings.Contains(m, "orin nano") || strings.Contains(m, "orin nx") {
		return false
	}
	return strings.Contains(m, "agx orin") || strings.Contains(m, "agxorin") || strings.Contains(m, "jetson-agx-orin")
}

// ClassifyDeviceTarget maps a HostDevice to "thor" or "orin" (only real-device verified hardware & L4T).
func ClassifyDeviceTarget(dev HostDevice) (string, error) {
	m := strings.ToLower(strings.TrimSpace(dev.Model + " " + dev.Platform))
	l4t := normalizeL4TVersion(dev.L4TVersion)

	if isThorModel(m) {
		if l4t == "" {
			return config.TargetThor, nil
		}
		if strings.HasPrefix(l4t, "38.4.") || strings.HasPrefix(l4t, "39.") {
			return config.TargetThor, nil
		}
		return "", fmt.Errorf("Thor 设备系统版本 %q 不受支持，仅支持 JetPack 7.1 (L4T 38.4.x) 与 JetPack 7.2+ (L4T 39.x)", dev.L4TVersion)
	}

	if isAGXOrinModel(m) {
		if l4t == "" {
			return config.TargetOrin, nil
		}
		if strings.HasPrefix(l4t, "39.") {
			return config.TargetOrin, nil
		}
		return "", fmt.Errorf("AGX Orin 设备系统版本 %q 不受支持，当前 SM87/CUDA 13.2 镜像仅支持 JetPack 7 (L4T 39.x)", dev.L4TVersion)
	}

	return "", fmt.Errorf("算力舱型号 %q 不受支持，当前版本仅支持已实机验收的 NVIDIA Jetson AGX Thor (T5000 / T4000) 与 AGX Orin 64GB (JetPack 7)", dev.Model)
}

func normalizeL4TVersion(raw string) string {
	v := strings.TrimSpace(strings.ToLower(raw))
	v = strings.TrimPrefix(v, "l4t")
	v = strings.TrimPrefix(v, "r")
	return strings.TrimSpace(v)
}
