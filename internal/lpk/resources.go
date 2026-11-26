package lpk

import (
	"fmt"
	"sort"
	"strings"

	"qwen35-9b-deployer/internal/config"
)

type MemoryBudget struct {
	Target                string
	TotalDeviceBytes      int64
	WeightsBytes          int64
	DraftBytes            int64
	RuntimeOverheadBytes  int64
	KVCacheBytes          int64
	StartupHeadroomBytes  int64
	SystemSafetyBytes     int64
	RequiredBytes         int64
	GPUMemoryUtilization  float64
	TotalCapacityTooSmall bool
	CapacityError         string
}

type memoryAssessment struct {
	Target                    string
	Resources                 HostDeviceResources
	Budget                    MemoryBudget
	AvailableBytes            int64
	ExistingReclaimableBytes  int64
	ProtectedReclaimableBytes int64
	ProjectedAvailableBytes   int64
	ShortfallBytes            int64
	ExistingStopIDs           []string
	ExistingStopReasons       []string
	ProtectedStopIDs          []string
	ProtectedStopReasons      []string
}

func (a memoryAssessment) totalCapacityInsufficient() bool {
	return a.Budget.TotalCapacityTooSmall
}

func (a memoryAssessment) memoryUnavailableWithoutModelStops() bool {
	if a.totalCapacityInsufficient() {
		return true
	}
	if a.ShortfallBytes <= 0 {
		return false
	}
	return len(a.ProtectedStopIDs) == 0 && len(a.ExistingStopIDs) == 0
}

func (a memoryAssessment) insufficientMemoryMessage() string {
	if a.Budget.CapacityError != "" {
		return a.Budget.CapacityError
	}
	return fmt.Sprintf("可用统一内存不足：当前 %s（释放冲突服务后预计 %s），此配置至少需要 %s",
		formatBytesGiB(a.AvailableBytes), formatBytesGiB(a.ProjectedAvailableBytes), formatBytesGiB(a.Budget.RequiredBytes))
}

// ComputeMemoryBudget calculates Qwen3.5-9B unified-memory requirements.
// Qwen3.5-9B uses a 3:1 hybrid architecture: 24 Gated DeltaNet linear-attention
// layers (fixed ~512 MiB recurrent state across 8 slots) and 8 full-attention
// layers (4 KV heads, head_dim=256), plus 1 embedded MTP layer.
func ComputeMemoryBudget(target string, totalDeviceBytes int64, contextLength int, manifest *LoadedManifests) MemoryBudget {
	var weightsBytes int64 = 8 << 30
	var draftBytes int64 = 0
	var kvPerTokenBytes int64 = 36864
	var ssmFixedBytes int64 = 768 << 20
	var runtimeOverhead int64 = 2500 << 20
	var headroom int64 = 2 << 30
	var sysSafety int64 = 4 << 30
	var gpuUtil float64 = 0.80

	if totalDeviceBytes >= 96<<30 {
		sysSafety = 8 << 30
	}

	if manifest != nil {
		if rt, ok := manifest.Runtimes.Targets[target]; ok {
			kvType := strings.ToLower(strings.TrimSpace(rt.LaunchProfile.KVCacheDtype))
			switch kvType {
			case "q8_0", "fp8":
				kvPerTokenBytes = 20000
			case "q4_0":
				kvPerTokenBytes = 11000
			}
		}
		var modelSum, draftSum int64
		for _, g := range manifest.GroupsForTarget(target) {
			var gBytes int64
			if len(g.Files) > 0 {
				for _, f := range g.Files {
					gBytes += f.Size
				}
			} else {
				gBytes = g.EstimatedBytes
			}
			if g.Role == "draft" {
				draftSum += gBytes
			} else {
				modelSum += gBytes
			}
		}
		if modelSum > 0 {
			weightsBytes = modelSum
		}
		draftBytes = draftSum
	}

	globalKVBytes := int64(contextLength) * kvPerTokenBytes
	kvBytes := ssmFixedBytes + globalKVBytes
	required := weightsBytes + draftBytes + runtimeOverhead + kvBytes + headroom

	b := MemoryBudget{
		Target:               target,
		TotalDeviceBytes:     totalDeviceBytes,
		WeightsBytes:         weightsBytes,
		DraftBytes:           draftBytes,
		RuntimeOverheadBytes: runtimeOverhead,
		KVCacheBytes:         kvBytes,
		StartupHeadroomBytes: headroom,
		SystemSafetyBytes:    sysSafety,
		RequiredBytes:        required,
		GPUMemoryUtilization: gpuUtil,
	}

	if totalDeviceBytes > 0 && totalDeviceBytes < MinSupportedTotalMemoryBytes {
		b.TotalCapacityTooSmall = true
		b.CapacityError = fmt.Sprintf("算力舱物理统一内存 %s 低于最低要求 %s", formatBytesGiB(totalDeviceBytes), formatBytesGiB(MinSupportedTotalMemoryBytes))
	} else if totalDeviceBytes > 0 && required+sysSafety > totalDeviceBytes {
		b.TotalCapacityTooSmall = true
		b.CapacityError = fmt.Sprintf("当前上下文 %d tokens 需要至少 %s 统一内存（含 %s 系统安全余量），超出算力舱总内存 %s，请调低上下文长度",
			contextLength, formatBytesGiB(required), formatBytesGiB(sysSafety), formatBytesGiB(totalDeviceBytes))
	}
	return b
}

// ComputeRequiredDiskBytes computes the peak disk space needed from the current state.
// Once artifacts/images are prepared on the device, completed temporary archive space
// is not double-charged during the pre-start check.
func ComputeRequiredDiskBytes(imageReady bool, missingModelBytes int64, rt RuntimeTargetManifest, cache *RuntimeCacheEntry) int64 {
	var required int64
	if missingModelBytes > 0 {
		required += missingModelBytes
	}
	if !imageReady {
		unpacked := rt.UnpackedSize
		if unpacked <= 0 {
			unpacked = 10 << 30
		}
		// Archive streamed to device + extracted image layers + transient unpack buffer
		required += rt.Size + 2*unpacked
	}
	if cache != nil && cache.Size > 0 {
		required += cache.Size * 2
	} else {
		// Cold JIT / CUDA Graph / Triton compilation workspace
		required += 2 << 30
	}
	// Safety margin
	required += 4 << 30
	return required
}

func assessMemoryAndProcesses(target string, resources HostDeviceResources, status HostProcessStatus, contextLength int, manifest *LoadedManifests) memoryAssessment {
	avail := resources.MemoryAvailableBytes
	if resources.VRAMAvailableBytes > avail {
		avail = resources.VRAMAvailableBytes
	}
	total := resources.MemoryTotalBytes
	if resources.VRAMTotalBytes > total {
		total = resources.VRAMTotalBytes
	}
	budget := ComputeMemoryBudget(target, total, contextLength, manifest)
	conflicts := classifyConflicts(status)

	var existingReclaim, protectedReclaim int64
	for _, id := range conflicts.existingIDs {
		if svc, ok := status.Services[id]; ok {
			existingReclaim += int64(svc.MemoryUsage)
		}
	}
	for _, id := range conflicts.protectedIDs {
		if svc, ok := status.Services[id]; ok {
			protectedReclaim += int64(svc.MemoryUsage)
		} else {
			for _, p := range status.OllamaPs {
				if strings.TrimSpace(p.PackageID) == id {
					protectedReclaim += int64(p.MemoryUsage)
				}
			}
		}
	}

	projected := avail + existingReclaim
	shortfall := budget.RequiredBytes - projected
	if shortfall < 0 {
		shortfall = 0
	}

	// Only treat protected processes as blocking when stopping our own service
	// isn't enough to satisfy RequiredBytes.
	var neededProtectedIDs, neededProtectedReasons []string
	if shortfall > 0 && len(conflicts.protectedIDs) > 0 {
		neededProtectedIDs = conflicts.protectedIDs
		neededProtectedReasons = conflicts.protectedReasons
		projected += protectedReclaim
	}

	return memoryAssessment{
		Target:                    target,
		Resources:                 resources,
		Budget:                    budget,
		AvailableBytes:            avail,
		ExistingReclaimableBytes:  existingReclaim,
		ProtectedReclaimableBytes: protectedReclaim,
		ProjectedAvailableBytes:   projected,
		ShortfallBytes:            shortfall,
		ExistingStopIDs:           conflicts.existingIDs,
		ExistingStopReasons:       conflicts.existingReasons,
		ProtectedStopIDs:          neededProtectedIDs,
		ProtectedStopReasons:      neededProtectedReasons,
	}
}

type classifiedConflicts struct {
	existingIDs      []string
	existingReasons  []string
	protectedIDs     []string
	protectedReasons []string
}

func classifyConflicts(status HostProcessStatus) classifiedConflicts {
	existing := map[string]struct{}{}
	protected := map[string]struct{}{}
	reasons := map[string]struct{}{}

	for _, p := range status.OllamaPs {
		if id := strings.TrimSpace(p.PackageID); id != "" {
			protected[id] = struct{}{}
			reasons["检测到正在运行的模型服务（可能中断现有推理请求）"] = struct{}{}
		}
	}
	for id, svc := range status.Services {
		trimmed := strings.TrimSpace(id)
		lower := strings.ToLower(trimmed)
		if trimmed == "" || isInfrastructureService(lower) {
			continue
		}
		if lower == strings.ToLower(config.DefaultServiceName) {
			existing[trimmed] = struct{}{}
			continue
		}
		if isConflictingModelService(lower, svc) {
			protected[trimmed] = struct{}{}
			reasons["检测到其他占内存模型服务（停止可能中断其请求）"] = struct{}{}
		}
	}

	var out classifiedConflicts
	for id := range existing {
		out.existingIDs = append(out.existingIDs, id)
	}
	for id := range protected {
		out.protectedIDs = append(out.protectedIDs, id)
	}
	for r := range reasons {
		out.protectedReasons = append(out.protectedReasons, r)
	}
	if len(out.existingIDs) > 0 {
		out.existingReasons = []string{"检测到已安装的 " + config.PublicServiceName + " 服务实例"}
	}
	sort.Strings(out.existingIDs)
	sort.Strings(out.protectedIDs)
	sort.Strings(out.protectedReasons)
	return out
}

func isInfrastructureService(lowerID string) bool {
	for _, infra := range []string{"traefik", "dozzle", "whoami", "serviceforward", "lzc-ai-agent", "jtop"} {
		if lowerID == infra || strings.HasPrefix(lowerID, infra+"-") || strings.HasPrefix(lowerID, infra+"_") {
			return true
		}
	}
	return false
}

func isConflictingModelService(lowerID string, svc HostServiceStatus) bool {
	if svc.MemoryUsage >= 1<<30 {
		return true
	}
	for _, kw := range []string{"vllm", "sglang", "qwen", "llama", "deepseek", "glm", "flint", "gemma", "minimax", "tensorfold", "decision", "ollama", "model"} {
		if strings.Contains(lowerID, kw) {
			return true
		}
	}
	return false
}

func formatBytesGiB(bytes int64) string {
	return fmt.Sprintf("%.1f GiB", float64(bytes)/(1<<30))
}
