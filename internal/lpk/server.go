package lpk

import (
	"context"
	"crypto/rand"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"qwen35-9b-deployer/internal/config"
)

//go:embed web/index.html
var indexHTML []byte

const defaultStateFile = "deployments.json"

type persistedState struct {
	Records       map[string]*persistedRecord   `json:"records"`
	PreservedData map[string]preservedDataEntry `json:"preservedData,omitempty"`
}

type persistedRecord struct {
	DeploymentRecord
	UserID              string   `json:"userId,omitempty"`
	StopExisting        bool     `json:"stopExisting,omitempty"`
	StopProtected       bool     `json:"stopProtected,omitempty"`
	ProtectedProcessIDs []string `json:"protectedProcessIds,omitempty"`
}

type preservedDataEntry struct {
	UserID       string    `json:"userId"`
	SerialNumber string    `json:"serialNumber"`
	Target       string    `json:"target"`
	ServiceID    string    `json:"serviceId"`
	PreservedAt  time.Time `json:"preservedAt"`
}

type Server struct {
	backendURL       string
	dataPath         string
	manifests        *LoadedManifests
	client           *http.Client
	stopClient       *http.Client
	deploymentClient *http.Client

	mu            sync.RWMutex
	admissionMu   sync.Mutex
	records       map[string]*DeploymentRecord
	preservedData map[string]preservedDataEntry
	devices       map[string][]Device
	cancelFuncs   map[string]context.CancelFunc
	workerDone    map[string]chan struct{}

	keyedMuGuard    sync.Mutex
	keyedLocks      map[string]chan struct{}
	deviceSwitchMu  map[string]chan struct{}
}

func NewServer(backendURL, dataDir string) (*Server, error) {
	if strings.TrimSpace(backendURL) == "" {
		backendURL = config.DefaultBackend
	}
	if strings.TrimSpace(dataDir) == "" {
		dataDir = os.Getenv("LZC_AGENT_DATA_DIR")
	}
	if strings.TrimSpace(dataDir) == "" {
		dataDir = os.Getenv("QWEN35_9B_DATA_DIR")
	}
	if strings.TrimSpace(dataDir) == "" {
		dataDir = "/lzcapp/var/qwen35-9b-defiant-fable"
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil, fmt.Errorf("create LPK data directory: %w", err)
	}
	manifests, err := LoadManifests()
	if err != nil {
		return nil, err
	}
	s := &Server{
		backendURL:       strings.TrimRight(backendURL, "/"),
		dataPath:         filepath.Join(dataDir, defaultStateFile),
		manifests:        manifests,
		client:           &http.Client{Timeout: defaultHTTPTimeout},
		stopClient:       &http.Client{Timeout: processStopTimeout},
		deploymentClient: &http.Client{Timeout: deploymentTimeout},
		records:          make(map[string]*DeploymentRecord),
		preservedData:    make(map[string]preservedDataEntry),
		devices:          make(map[string][]Device),
		cancelFuncs:      make(map[string]context.CancelFunc),
		workerDone:       make(map[string]chan struct{}),
		keyedLocks:       make(map[string]chan struct{}),
		deviceSwitchMu:   make(map[string]chan struct{}),
	}
	if err := s.loadState(); err != nil {
		return nil, err
	}
	return s, nil
}

func (s *Server) Close() {
	s.mu.Lock()
	for _, cancel := range s.cancelFuncs {
		cancel()
	}
	_ = s.flushLocked()
	s.mu.Unlock()
}

func (s *Server) acquireKeyedLock(ctx context.Context, key string) func() {
	s.keyedMuGuard.Lock()
	ch, ok := s.keyedLocks[key]
	if !ok {
		ch = make(chan struct{}, 1)
		ch <- struct{}{}
		s.keyedLocks[key] = ch
	}
	s.keyedMuGuard.Unlock()

	select {
	case <-ctx.Done():
		return nil
	case <-ch:
		return func() { ch <- struct{}{} }
	}
}

func (s *Server) acquireDeviceSwitchLock(ctx context.Context, serial string) func() {
	s.keyedMuGuard.Lock()
	ch, ok := s.deviceSwitchMu[serial]
	if !ok {
		ch = make(chan struct{}, 1)
		ch <- struct{}{}
		s.deviceSwitchMu[serial] = ch
	}
	s.keyedMuGuard.Unlock()

	select {
	case <-ctx.Done():
		return nil
	case <-ch:
		return func() { ch <- struct{}{} }
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/health", s.handleHealth)
	mux.HandleFunc("/api/v1/descriptor", s.handleDescriptor)
	mux.HandleFunc("/api/v1/devices", s.handleDevices)
	mux.HandleFunc("/api/v1/deployments/requirements", s.handleRequirements)
	mux.HandleFunc("/api/v1/deployments", s.handleDeployments)
	mux.HandleFunc("/api/v1/deployments/", s.handleDeploymentItem)
	mux.HandleFunc("/", s.handleStatic)
	return withJSONHeaders(mux)
}

func withJSONHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if origin := strings.TrimSpace(r.Header.Get("Origin")); origin != "" {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Vary", "Origin")
		} else {
			w.Header().Set("Access-Control-Allow-Origin", "*")
		}
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-HC-User-ID")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleStatic(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write(indexHTML)
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":              true,
		"package":         config.PackageID,
		"version":         config.Version,
		"protocolVersion": ProtocolVersion,
	})
}

func (s *Server) handleDescriptor(w http.ResponseWriter, _ *http.Request) {
	thorRT, thorArchive, thorErr := s.manifests.RuntimeImageArtifactForTarget(config.TargetThor, config.DefaultModelProvider)
	var defaultRT RuntimeImageDescriptor
	var artifacts = map[string]ArtifactGroup{}
	var runtimeImages = map[string]RuntimeImageDescriptor{}

	if thorErr == nil {
		defaultRT = toRuntimeImageDescriptor(thorRT)
		artifacts["runtime-image"] = ArtifactGroup{
			Name:       "Qwen 3.5 9B Defiant Fable 运行镜像",
			Repository: thorRT.Sources[config.DefaultModelProvider].Repository,
			Files:      []ArtifactFile{thorArchive},
		}
	} else {
		rawRT := s.manifests.Runtimes.Targets[config.TargetThor]
		defaultRT = toRuntimeImageDescriptor(rawRT)
	}
	for _, target := range []string{config.TargetThor, config.TargetOrin} {
		if rt, ok := s.manifests.Runtimes.Targets[target]; ok {
			runtimeImages[target] = toRuntimeImageDescriptor(rt)
		}
	}

	var modelEntries []ModelSourceEntry
	var totalModelSize int64
	for _, g := range s.manifests.Models.Models {
		var sz int64
		for _, f := range g.Files {
			sz += f.Size
		}
		if sz == 0 {
			sz = g.EstimatedBytes
		}
		totalModelSize += sz
		providers := map[string]ProviderSource{}
		for pName, pin := range g.Sources {
			providers[pName] = ProviderSource{Repository: pin.Repository, Revision: pin.Revision}
		}
		modelEntries = append(modelEntries, ModelSourceEntry{
			Role:        g.Role,
			Name:        g.Name,
			Destination: g.Destination,
			FileCount:   len(g.Files),
			Size:        sz,
			Providers:   providers,
		})
	}

	desc := Descriptor{
		ProtocolVersion:        ProtocolVersion,
		MinHostProtocolVersion: MinHostProtocol,
		Package:                config.PackageID,
		Name:                   config.PublicServiceName,
		Model: ModelDescriptor{
			Name:               config.PublicServiceName,
			TaskType:           "chat",
			InputMode:          "multimodal",
			OutputType:         "text",
			ContextLength:      config.DefaultContextLength,
			MinContextLength:   config.MinContextLength,
			MaxContextLength:   config.MaxContextLength,
			SafeReserveBytes:   4 << 30,
			MaxConcurrency:     config.MaxConcurrency,
			NetworkDrivePath:   config.NetworkDriveDisplay,
			ReuseAcrossDevices: true,
		},
		Parameters: []ParameterDescriptor{
			{
				ID:          "contextLength",
				Label:       "上下文长度",
				Type:        "integer",
				Unit:        "tokens",
				Default:     config.DefaultContextLength,
				Min:         config.MinContextLength,
				Max:         config.MaxContextLength,
				Step:        config.ContextLengthStep,
				Fixed:       false,
				Description: "支持 16K–256K 原生上下文，默认 128K（1024 tokens 步长）",
			},
			{
				ID:      "modelProvider",
				Label:   "下载源",
				Type:    "select",
				Default: config.DefaultModelProvider,
				Fixed:   false,
				Options: []ParameterOptionDescriptor{
					{Value: "modelscope", Label: "ModelScope"},
					{Value: "huggingface", Label: "Hugging Face"},
				},
			},
		},
		SupportedDevices: []string{"AGX Thor (T5000 / Dev / T4000)", "AGX Orin 64GB (JetPack 7)"},
		Capabilities: map[string]bool{
			"customContextLength":      true,
			"customConcurrency":        false,
			"descriptorParameters":     true,
			"prebuiltImages":           true,
			"networkDriveReuse":        true,
			"unifiedProgress":          true,
			"cancelDeployment":         true,
			"piAgentConfig":            true,
			"runtimeModelDownload":     true,
			"modelSourceSelection":     true,
			"selectiveProcessStop":     true,
			"stopExistingConfirmation": true,
			"restartStopped":           true,
			"ociRuntimePull":           false,
			"delete":                   true,
			"persistentData":           true,
			"managedUninstall":         true,
		},
		Artifacts:     artifacts,
		RuntimeImage:  defaultRT,
		RuntimeImages: runtimeImages,
		ModelSources: &ModelSourcesDescriptor{
			Policy:          "user-selected",
			Detection:       "installation-ui",
			RegionProviders: map[string]string{"CN": "modelscope", "default": "huggingface"},
			Integrity:       "size-and-sha256",
			Cache:           "device-persistent",
			TotalSize:       totalModelSize,
			Models:          modelEntries,
		},
		Modes: []ModeDescriptor{
			{
				ID:          "custom",
				Title:       "标准部署配置",
				MaxNumSeqs:  config.MaxConcurrency,
				Description: "最大 8 路并发共享 KV 池，支持 32K–256K 上下文与图文多模态输入。",
				Pros:        []string{"按硬件自动选择专属量化权重与运行环境", "支持文本与图像多模态输入"},
				Cons:        []string{"单设备最高受统一内存与上下文 KV 容量约束"},
				Image:       defaultRT.Image,
			},
		},
	}
	writeJSON(w, http.StatusOK, desc)
}

func toRuntimeImageDescriptor(rt RuntimeTargetManifest) RuntimeImageDescriptor {
	providers := map[string]ProviderSource{}
	var canonicalURL string
	for pName, src := range rt.Sources {
		providers[pName] = ProviderSource{
			Repository: src.Repository,
			Revision:   src.Revision,
			Path:       src.Path,
		}
		if pName == config.DefaultModelProvider && src.Revision != "" {
			canonicalURL = BuildRuntimeArchiveURL(pName, src)
		}
	}
	return RuntimeImageDescriptor{
		Source:             "archive",
		Name:               rt.Name,
		Image:              rt.Image,
		ImageID:            rt.ImageID,
		ConfigImageID:      rt.ConfigImageID,
		RuntimeImageSHA256: rt.RuntimeImageSHA256,
		URL:                canonicalURL,
		Providers:          providers,
		Size:               rt.Size,
		SHA256:             rt.SHA256,
	}
}

func (s *Server) handleDevices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	userID := requestUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, errors.New("需要用户身份标识"))
		return
	}
	hosts, err := s.fetchHostDevices(r.Context(), userID)
	if err != nil {
		s.mu.RLock()
		cached := append([]Device(nil), s.devices[userID]...)
		s.mu.RUnlock()
		if len(cached) == 0 {
			writeError(w, http.StatusBadGateway, err)
			return
		}
		writeJSON(w, http.StatusOK, cached)
		return
	}

	out := make([]Device, 0, len(hosts))
	for _, h := range hosts {
		if !IsSupportedHardwareModel(h.Model, h.Platform) {
			continue
		}
		target, targetErr := ClassifyDeviceTarget(h)
		lastErr := h.LastError
		if targetErr != nil && lastErr == "" {
			lastErr = targetErr.Error()
		}
		out = append(out, Device{
			SerialNumber: h.SerialNumber,
			Name:         h.Name,
			Model:        h.Model,
			L4TVersion:   h.L4TVersion,
			Platform:     h.Platform,
			Target:       target,
			IP:           h.IP,
			Port:         config.DefaultAPIPort,
			AgentPort:    h.AgentPort,
			Online:       h.Online,
			ModelReady:   s.hasReadyOrPreservedDeployment(userID, h.SerialNumber),
			LastError:    lastErr,
		})
	}
	s.mu.Lock()
	s.devices[userID] = append([]Device(nil), out...)
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) hasReadyOrPreservedDeployment(userID, serial string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, r := range s.records {
		if r.UserID == userID && r.SerialNumber == serial && (r.Status == "succeeded" || r.Status == "stopped") {
			return true
		}
	}
	_, preserved := s.preservedData[preservedKey(userID, serial)]
	return preserved
}

func (s *Server) handleRequirements(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	userID := requestUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, errors.New("需要用户身份标识"))
		return
	}
	serial := strings.TrimSpace(r.URL.Query().Get("serialNumber"))
	if serial == "" {
		writeError(w, http.StatusBadRequest, errors.New("serialNumber 不能为空"))
		return
	}
	settings, err := ParseRequirementsParams(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	dev, target, err := s.resolveHostDeviceAndTarget(r.Context(), userID, serial)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"online":               false,
			"requiresStop":         false,
			"requiresConfirmation": false,
			"blocked":              false,
			"error":                err.Error(),
			"resourceError":        err.Error(),
			"contextLength":        settings.ContextLength,
			"maxConcurrency":       config.MaxConcurrency,
		})
		return
	}
	if !dev.Online {
		msg := fmt.Sprintf("算力舱 %s 当前离线", dev.Name)
		writeJSON(w, http.StatusOK, map[string]any{
			"online":               false,
			"requiresStop":         false,
			"requiresConfirmation": false,
			"blocked":              false,
			"error":                msg,
			"resourceError":        msg,
			"contextLength":        settings.ContextLength,
			"maxConcurrency":       config.MaxConcurrency,
		})
		return
	}

	res, err := s.fetchDeviceResources(r.Context(), serial, userID)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"online":         false,
			"error":          err.Error(),
			"resourceError":  err.Error(),
			"contextLength":  settings.ContextLength,
			"maxConcurrency": config.MaxConcurrency,
		})
		return
	}
	proc, err := s.fetchProcessStatus(r.Context(), serial, userID)
	if err != nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"online":         false,
			"error":          err.Error(),
			"resourceError":  err.Error(),
			"contextLength":  settings.ContextLength,
			"maxConcurrency": config.MaxConcurrency,
		})
		return
	}

	assessment := assessMemoryAndProcesses(target, res, proc, settings.ContextLength, s.manifests)
	resp := map[string]any{
		"online":                   true,
		"target":                   target,
		"contextLength":            settings.ContextLength,
		"maxConcurrency":           config.MaxConcurrency,
		"memoryTotalBytes":         res.MemoryTotalBytes,
		"memoryAvailableBytes":     assessment.AvailableBytes,
		"requiredMemoryBytes":      assessment.Budget.RequiredBytes,
		"projectedAvailableBytes":  assessment.ProjectedAvailableBytes,
		"requiresStop":             len(assessment.ExistingStopIDs) > 0,
		"reasons":                  assessment.ExistingStopReasons,
		"requiresConfirmation":     false,
		"blocked":                  false,
		"insufficientMemory":       false,
		"insufficientDisk":         false,
	}

	// Check disk admission peak when manifests are locked; still check minimum baseline otherwise.
	if rt, ok := s.manifests.Runtimes.Targets[target]; ok {
		imageReady := false
		if rdy, _, imgErr := s.deviceRuntimeImageReady(r.Context(), serial, userID, rt); imgErr == nil {
			imageReady = rdy
		}
		missingModel := s.manifests.ModelBytesForTarget(target)
		if err := s.manifests.ValidateTargetLocked(target); err == nil {
			if mStat, arts, tot, mErr := s.fetchDeviceModelsStatus(r.Context(), serial, userID, target, settings.ModelProvider, false); mErr == nil {
				missingModel = missingBytesFromStatus(mStat, arts, tot)
			}
		}
		reqDisk := ComputeRequiredDiskBytes(imageReady, missingModel, rt, nil)
		resp["requiredDiskBytes"] = reqDisk
		resp["availableDiskBytes"] = res.DiskAvailableBytes
		if res.DiskAvailableKnown && res.DiskAvailableBytes < reqDisk {
			resp["insufficientDisk"] = true
			resp["resourceError"] = fmt.Sprintf("算力舱磁盘空间不足：峰值至少需 %s，当前可用 %s",
				formatBytesGiB(reqDisk), formatBytesGiB(res.DiskAvailableBytes))
		}
	}

	if assessment.totalCapacityInsufficient() || assessment.memoryUnavailableWithoutModelStops() {
		resp["insufficientMemory"] = true
		resp["resourceError"] = assessment.insufficientMemoryMessage()
		resp["requiresConfirmation"] = false
		writeJSON(w, http.StatusOK, resp)
		return
	}

	if len(assessment.ProtectedStopIDs) > 0 {
		resp["blocked"] = true
		resp["requiresConfirmation"] = true
		resp["insufficientMemory"] = true
		resp["blockReasons"] = assessment.ProtectedStopReasons
		resp["protectedProcessIds"] = assessment.ProtectedStopIDs
	} else if len(assessment.ExistingStopIDs) > 0 {
		resp["requiresConfirmation"] = true
	}

	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleDeployments(w http.ResponseWriter, r *http.Request) {
	userID := requestUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, errors.New("需要用户身份标识"))
		return
	}

	switch r.Method {
	case http.MethodGet:
		s.mu.RLock()
		items := make([]DeploymentRecord, 0, len(s.records))
		for _, rec := range s.records {
			if rec.UserID == userID {
				items = append(items, cloneRecord(rec))
			}
		}
		s.mu.RUnlock()
		sort.Slice(items, func(i, j int) bool {
			return items[i].StartedAt.After(items[j].StartedAt)
		})
		writeJSON(w, http.StatusOK, items)

	case http.MethodPost:
		var req deployRequest
		if err := decodeJSON(r, &req); err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		settings, err := NormalizeDeployRequest(req)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		serials, err := normalizeSerialList(req)
		if err != nil {
			writeError(w, http.StatusBadRequest, err)
			return
		}
		approvedSet := make(map[string]bool, len(req.ProtectedProcessIDs))
		for _, pid := range req.ProtectedProcessIDs {
			pid = strings.TrimSpace(pid)
			if pid != "" {
				approvedSet[pid] = true
			}
		}

		// Read-only preflight across all selected serials BEFORE queueing or stopping anything.
		type checkedDevice struct {
			host       HostDevice
			target     string
			assessment memoryAssessment
		}
		checked := make([]checkedDevice, 0, len(serials))
		var blockingIDs []string
		var blockReasons []string
		var stopReasons []string
		var capacityErrors []string
		var diskErrors []string
		approvalMismatch := false

		for _, serial := range serials {
			host, target, err := s.resolveHostDeviceAndTarget(r.Context(), userID, serial)
			if err != nil {
				writeJSON(w, http.StatusConflict, map[string]any{
					"error":                err.Error(),
					"requiresConfirmation": false,
				})
				return
			}
			if !host.Online {
				writeJSON(w, http.StatusConflict, map[string]any{
					"error":                fmt.Sprintf("算力舱 %s 当前离线", host.Name),
					"requiresConfirmation": false,
				})
				return
			}
			if err := s.manifests.ValidateTargetLocked(target); err != nil {
				writeJSON(w, http.StatusConflict, map[string]any{
					"error":                err.Error(),
					"requiresConfirmation": false,
				})
				return
			}
			res, err := s.fetchDeviceResources(r.Context(), serial, userID)
			if err != nil {
				writeError(w, http.StatusBadGateway, err)
				return
			}
			proc, err := s.fetchProcessStatus(r.Context(), serial, userID)
			if err != nil {
				writeError(w, http.StatusBadGateway, err)
				return
			}
			rt := s.manifests.Runtimes.Targets[target]
			imageReady, _, _ := s.deviceRuntimeImageReady(r.Context(), serial, userID, rt)
			missingModel := s.manifests.ModelBytesForTarget(target)
			if mStat, arts, tot, mErr := s.fetchDeviceModelsStatus(r.Context(), serial, userID, target, settings.ModelProvider, false); mErr == nil {
				missingModel = missingBytesFromStatus(mStat, arts, tot)
			}
			reqDisk := ComputeRequiredDiskBytes(imageReady, missingModel, rt, nil)
			if res.DiskAvailableKnown && res.DiskAvailableBytes < reqDisk {
				diskErrors = append(diskErrors, fmt.Sprintf("%s：可用磁盘 %s 低于峰值需求 %s", host.Name, formatBytesGiB(res.DiskAvailableBytes), formatBytesGiB(reqDisk)))
			}

			a := assessMemoryAndProcesses(target, res, proc, settings.ContextLength, s.manifests)
			if a.memoryUnavailableWithoutModelStops() {
				capacityErrors = append(capacityErrors, fmt.Sprintf("%s：%s", host.Name, a.insufficientMemoryMessage()))
			}
			if len(a.ProtectedStopIDs) > 0 {
				for _, pid := range a.ProtectedStopIDs {
					blockingIDs = append(blockingIDs, pid)
					if !req.StopProtected || !approvedSet[pid] {
						approvalMismatch = true
					}
				}
				for _, rsn := range a.ProtectedStopReasons {
					blockReasons = append(blockReasons, fmt.Sprintf("%s：%s", host.Name, rsn))
				}
			}
			if len(a.ExistingStopIDs) > 0 && !req.StopExisting {
				for _, rsn := range a.ExistingStopReasons {
					stopReasons = append(stopReasons, fmt.Sprintf("%s：%s", host.Name, rsn))
				}
			}
			checked = append(checked, checkedDevice{host: host, target: target, assessment: a})
		}

		if len(diskErrors) > 0 {
			writeJSON(w, http.StatusConflict, map[string]any{
				"error":                strings.Join(diskErrors, "；"),
				"insufficientDisk":     true,
				"requiresConfirmation": false,
			})
			return
		}
		if len(capacityErrors) > 0 {
			writeJSON(w, http.StatusConflict, map[string]any{
				"error":                strings.Join(capacityErrors, "；"),
				"insufficientMemory":   true,
				"requiresConfirmation": false,
			})
			return
		}
		blockingIDs = dedupeAndSort(blockingIDs)
		if len(blockingIDs) > 0 && (!req.StopProtected || approvalMismatch) {
			writeJSON(w, http.StatusConflict, map[string]any{
				"blocked":              true,
				"requiresConfirmation": true,
				"requiresStop":         len(stopReasons) > 0,
				"insufficientMemory":   true,
				"error":                "算力舱统一内存不足；模型文件和镜像准备完成后，仅在启动时按需停止已确认的冲突服务，届时可能中断现有请求。",
				"blockReasons":         blockReasons,
				"reasons":              append(stopReasons, blockReasons...),
				"protectedProcessIds":  blockingIDs,
				"serialNumbers":        serials,
			})
			return
		}
		if len(stopReasons) > 0 {
			writeJSON(w, http.StatusConflict, map[string]any{
				"requiresConfirmation": true,
				"requiresStop":         true,
				"error":                "检测到已运行的同类模型服务，准备完成后将在启动前停止旧实例。",
				"reasons":              stopReasons,
				"serialNumbers":        serials,
			})
			return
		}

		s.admissionMu.Lock()
		defer s.admissionMu.Unlock()

		s.mu.Lock()
		for _, item := range checked {
			for _, rec := range s.records {
				if rec.UserID == userID && rec.SerialNumber == item.host.SerialNumber && isRecordActive(rec.Status) {
					s.mu.Unlock()
					writeJSON(w, http.StatusConflict, map[string]any{
						"error":        "该算力舱已有部署任务正在运行，请等待完成后再操作",
						"deploymentId": rec.ID,
					})
					return
				}
			}
		}

		var outRecords []*DeploymentRecord
		for _, item := range checked {
			rt := s.manifests.Runtimes.Targets[item.target]
			rec := s.reusableRecordLocked(userID, item.host.SerialNumber)
			if rec == nil {
				id, idErr := newRandomID()
				if idErr != nil {
					s.mu.Unlock()
					writeError(w, http.StatusInternalServerError, idErr)
					return
				}
				rec = &DeploymentRecord{ID: id}
				s.records[id] = rec
			}
			now := time.Now().UTC()
			*rec = DeploymentRecord{
				ID:                   rec.ID,
				ModelName:            config.PublicServiceName,
				ModelVersion:         config.Version,
				ModelID:              config.PublicModelID,
				APIKey:               config.DefaultAPIKey,
				SerialNumber:         item.host.SerialNumber,
				DeviceName:           item.host.Name,
				DeviceModel:          item.host.Model,
				RuntimeProfile:       item.target,
				Mode:                 "custom",
				MaxNumSeqs:           config.MaxConcurrency,
				Concurrency:          config.MaxConcurrency,
				ContextLength:        settings.ContextLength,
				ModelProvider:        settings.ModelProvider,
				Parameters:           map[string]any{"contextLength": settings.ContextLength, "modelProvider": settings.ModelProvider},
				EstimatedMemoryBytes: item.assessment.Budget.RequiredBytes,
				RemainingMemoryBytes: maxInt64(0, item.assessment.ProjectedAvailableBytes-item.assessment.Budget.RequiredBytes),
				MaxConcurrency:       config.MaxConcurrency,
				Image:                rt.Image,
				Status:               "queued",
				Stage:                "queued",
				Message:              "等待准备模型制品与运行环境",
				StartedAt:            now,
				Progress:             0,
				PhaseProgress:        0,
				Accessible:           false,
				Logs:                 []DeploymentLog{{Timestamp: now, Message: "部署任务已受理，正在排队准备制品"}},
				StopExisting:         req.StopExisting,
				StopProtected:        req.StopProtected,
				ProtectedProcessIDs:  append([]string(nil), req.ProtectedProcessIDs...),
				UserID:               userID,
			}
			outRecords = append(outRecords, rec)
		}
		if err := s.flushLocked(); err != nil {
			s.mu.Unlock()
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		respRecords := make([]DeploymentRecord, len(outRecords))
		for i, r := range outRecords {
			respRecords[i] = cloneRecord(r)
		}
		s.mu.Unlock()

		for _, r := range respRecords {
			go s.runDeploymentWorker(r.ID)
		}
		if len(respRecords) == 1 {
			writeJSON(w, http.StatusAccepted, respRecords[0])
		} else {
			writeJSON(w, http.StatusAccepted, map[string]any{"records": respRecords})
		}

	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleDeploymentItem(w http.ResponseWriter, r *http.Request) {
	userID := requestUserID(r)
	if userID == "" {
		writeError(w, http.StatusUnauthorized, errors.New("需要用户身份标识"))
		return
	}
	trimmed := strings.TrimPrefix(r.URL.Path, "/api/v1/deployments/")
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		w.WriteHeader(http.StatusNotFound)
		return
	}
	id := parts[0]
	if len(parts) == 1 {
		switch r.Method {
		case http.MethodGet:
			s.mu.RLock()
			rec, ok := s.records[id]
			if !ok || rec.UserID != userID {
				s.mu.RUnlock()
				writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
				return
			}
			cp := cloneRecord(rec)
			s.mu.RUnlock()
			writeJSON(w, http.StatusOK, cp)
		case http.MethodPatch:
			s.handlePatchDeployment(w, r, id, userID)
		case http.MethodDelete:
			s.handleDeleteDeployment(w, r, id, userID)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
		return
	}
	if len(parts) == 2 {
		switch parts[1] {
		case "cancel":
			s.handleCancelDeployment(w, r, id, userID)
		case "retry":
			s.handleRetryDeployment(w, r, id, userID)
		case "stop":
			s.handleStopDeployment(w, r, id, userID)
		case "start":
			s.handleStartDeployment(w, r, id, userID)
		case "pi-agent-config":
			s.handlePiAgentConfig(w, r, id, userID)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
		return
	}
	w.WriteHeader(http.StatusNotFound)
}

func (s *Server) handlePatchDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	var body struct {
		DeviceName string `json:"deviceName"`
	}
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	name := strings.TrimSpace(body.DeviceName)
	if name == "" || len([]rune(name)) > 80 {
		writeError(w, http.StatusBadRequest, errors.New("设备展示名称不能为空且不能超过 80 个字符"))
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	old := rec.DeviceName
	rec.DeviceName = name
	if err := s.flushLocked(); err != nil {
		rec.DeviceName = old
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	writeJSON(w, http.StatusOK, cloneRecord(rec))
}

// handleDeleteDeployment implements the §4.2 persistentData deletion contract:
// - deleteData omitted or false: cancel active task safely, stop service via selective /process/stop, preserve persistent data, hide record.
// - deleteData=true: cancel active task safely, execute destructive DELETE /services/{serviceId}, remove preserved metadata & record.
// - invalid or repeated deleteData: HTTP 400 with no side effects.
func (s *Server) handleDeleteDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	deleteData, err := ParseDeleteDataParam(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}

	s.mu.Lock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	cancel := s.cancelFuncs[id]
	doneCh := s.workerDone[id]
	serial := rec.SerialNumber
	target := rec.RuntimeProfile
	status := rec.Status
	s.mu.Unlock()

	if cancel != nil {
		cancel()
		if doneCh != nil {
			select {
			case <-doneCh:
			case <-time.After(15 * time.Second):
				writeError(w, http.StatusConflict, errors.New("正在取消活动部署任务，请稍后重试删除"))
				return
			}
		}
	}

	if deleteData {
		resp, reqErr := s.doJSON(r.Context(), http.MethodDelete, s.hostPath("/devices/"+url.PathEscape(serial)+"/services/"+url.PathEscape(config.DefaultServiceName)), nil, userID)
		if reqErr != nil {
			writeError(w, http.StatusBadGateway, fmt.Errorf("清理算力舱服务与持久化数据失败: %w", reqErr))
			return
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		resp.Body.Close()
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			// Tolerate missing service directory when already cleaned or never created.
			lowerBody := strings.ToLower(string(body))
			if !strings.Contains(lowerBody, "not found") && !strings.Contains(lowerBody, "not_found") {
				writeError(w, http.StatusBadGateway, fmt.Errorf("清理算力舱服务与持久化数据失败: %s", hostResponseError(resp.Status, body)))
				return
			}
		}
	} else if status == "succeeded" || status == "starting" {
		if err := s.requireSelectiveProcessStop(r.Context(), userID); err == nil {
			_ = s.postHostJSONWithClient(r.Context(), s.stopClient, "/devices/"+url.PathEscape(serial)+"/process/stop", ProcessStopRequest{
				PackageIDs: []string{config.DefaultServiceName},
			}, userID, nil)
		}
	}

	s.mu.Lock()
	delete(s.records, id)
	delete(s.cancelFuncs, id)
	delete(s.workerDone, id)
	pKey := preservedKey(userID, serial)
	if deleteData {
		delete(s.preservedData, pKey)
	} else {
		s.preservedData[pKey] = preservedDataEntry{
			UserID:       userID,
			SerialNumber: serial,
			Target:       target,
			ServiceID:    config.DefaultServiceName,
			PreservedAt:  time.Now().UTC(),
		}
	}
	if err := s.flushLocked(); err != nil {
		s.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]any{"deleted": true})
}

func (s *Server) handleCancelDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.mu.Lock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	if isRecordActive(rec.Status) {
		rec.Status = "canceled"
		rec.Stage = "canceled"
		rec.Message = "部署已取消"
		rec.FinishedAt = time.Now().UTC()
		rec.ElapsedSeconds = rec.FinishedAt.Sub(rec.StartedAt).Seconds()
		rec.Logs = append(rec.Logs, DeploymentLog{Timestamp: rec.FinishedAt, Message: "用户已取消部署任务"})
		cancel := s.cancelFuncs[id]
		_ = s.flushLocked()
		cp := cloneRecord(rec)
		s.mu.Unlock()
		if cancel != nil {
			cancel()
		}
		writeJSON(w, http.StatusOK, cp)
		return
	}
	cp := cloneRecord(rec)
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, cp)
}

func (s *Server) handleRetryDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()

	s.mu.Lock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	if rec.Status != "failed" {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, errors.New("仅支持重试处于失败状态的部署记录"))
		return
	}
	cp := cloneRecord(rec)
	s.mu.Unlock()

	settings, err := NormalizeDeployRequest(deployRequest{
		ContextLength: &cp.ContextLength,
		ModelProvider: cp.ModelProvider,
		Parameters:    cp.Parameters,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	host, target, err := s.resolveHostDeviceAndTarget(r.Context(), userID, cp.SerialNumber)
	if err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}
	if !host.Online {
		writeError(w, http.StatusConflict, fmt.Errorf("算力舱 %s 当前离线", host.Name))
		return
	}
	if err := s.manifests.ValidateTargetLocked(target); err != nil {
		writeError(w, http.StatusConflict, err)
		return
	}

	s.mu.Lock()
	rec = s.records[id]
	if rec == nil || rec.Status != "failed" {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, errors.New("部署记录状态已变更"))
		return
	}
	now := time.Now().UTC()
	rec.RuntimeProfile = target
	rec.ContextLength = settings.ContextLength
	rec.ModelProvider = settings.ModelProvider
	rec.Parameters = map[string]any{"contextLength": settings.ContextLength, "modelProvider": settings.ModelProvider}
	rec.Status = "queued"
	rec.Stage = "queued"
	rec.Message = "重试部署（保留已下载与断点文件）"
	rec.Error = ""
	rec.Progress = 0
	rec.PhaseProgress = 0
	rec.Accessible = false
	rec.StartedAt = now
	rec.FinishedAt = time.Time{}
	rec.Logs = append(rec.Logs, DeploymentLog{Timestamp: now, Message: "开始重试部署，复用已有制品进度"})
	if err := s.flushLocked(); err != nil {
		s.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	out := cloneRecord(rec)
	s.mu.Unlock()

	go s.runDeploymentWorker(id)
	writeJSON(w, http.StatusAccepted, out)
}

func (s *Server) handleStopDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.mu.RLock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.RUnlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	if rec.Status != "succeeded" && rec.Status != "stopped" {
		s.mu.RUnlock()
		writeError(w, http.StatusConflict, errors.New("只有已部署或已停止的实例可执行停止"))
		return
	}
	serial := rec.SerialNumber
	s.mu.RUnlock()

	if err := s.requireSelectiveProcessStop(r.Context(), userID); err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	if err := s.postHostJSONWithClient(r.Context(), s.stopClient, "/devices/"+url.PathEscape(serial)+"/process/stop", ProcessStopRequest{
		PackageIDs: []string{config.DefaultServiceName},
	}, userID, nil); err != nil {
		writeError(w, http.StatusBadGateway, fmt.Errorf("停止模型服务失败: %w", err))
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if cur, exists := s.records[id]; exists && cur.UserID == userID {
		cur.Status = "stopped"
		cur.Stage = "stopped"
		cur.Message = "模型服务已停止（模型文件与镜像已保留）"
		cur.Accessible = false
		cur.MemoryUsageBytes = 0
		cur.FinishedAt = time.Now().UTC()
		cur.ElapsedSeconds = cur.FinishedAt.Sub(cur.StartedAt).Seconds()
		cur.Logs = append(cur.Logs, DeploymentLog{Timestamp: cur.FinishedAt, Message: cur.Message})
		_ = s.flushLocked()
		writeJSON(w, http.StatusOK, cloneRecord(cur))
		return
	}
	writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
}

func (s *Server) handleStartDeployment(w http.ResponseWriter, r *http.Request, id, userID string) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	var req startServiceRequest
	if r.Body != nil {
		dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil && !errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, err)
			return
		}
	}

	s.admissionMu.Lock()
	defer s.admissionMu.Unlock()

	s.mu.Lock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	if rec.Status == "succeeded" || rec.Status == "starting" {
		cp := cloneRecord(rec)
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, cp)
		return
	}
	if rec.Status != "stopped" {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, errors.New("只有已停止的模型服务可以重新启动"))
		return
	}
	cp := cloneRecord(rec)
	s.mu.Unlock()

	settings, err := NormalizeDeployRequest(deployRequest{
		ContextLength: &cp.ContextLength,
		ModelProvider: cp.ModelProvider,
		Parameters:    cp.Parameters,
	})
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	host, target, err := s.resolveHostDeviceAndTarget(r.Context(), userID, cp.SerialNumber)
	if err != nil || !host.Online {
		writeError(w, http.StatusConflict, errors.New("算力舱当前离线或不再受支持"))
		return
	}
	res, err := s.fetchDeviceResources(r.Context(), cp.SerialNumber, userID)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	proc, err := s.fetchProcessStatus(r.Context(), cp.SerialNumber, userID)
	if err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}

	assessment := assessMemoryAndProcesses(target, res, proc, settings.ContextLength, s.manifests)
	if assessment.memoryUnavailableWithoutModelStops() {
		writeJSON(w, http.StatusConflict, map[string]any{
			"insufficientMemory":   true,
			"requiresConfirmation": false,
			"error":                assessment.insufficientMemoryMessage(),
		})
		return
	}

	approved := map[string]bool{}
	if req.StopProtected {
		for _, pid := range req.ProtectedProcessIDs {
			approved[strings.TrimSpace(pid)] = true
		}
	}
	approvalMismatch := false
	for _, pid := range assessment.ProtectedStopIDs {
		if !req.StopProtected || !approved[pid] {
			approvalMismatch = true
		}
	}
	if len(assessment.ProtectedStopIDs) > 0 && (!req.StopProtected || approvalMismatch) {
		writeJSON(w, http.StatusConflict, map[string]any{
			"blocked":              true,
			"requiresConfirmation": true,
			"requiresStop":         len(assessment.ExistingStopIDs) > 0,
			"insufficientMemory":   true,
			"error":                "算力舱统一内存不足；请确认停止其他冲突服务后再启动。",
			"blockReasons":         assessment.ProtectedStopReasons,
			"protectedProcessIds":  assessment.ProtectedStopIDs,
		})
		return
	}
	if len(assessment.ExistingStopIDs) > 0 && !req.StopExisting {
		writeJSON(w, http.StatusConflict, map[string]any{
			"requiresConfirmation": true,
			"requiresStop":         true,
			"error":                "检测到同类服务实例，请确认停止后再启动。",
			"reasons":              assessment.ExistingStopReasons,
		})
		return
	}

	s.mu.Lock()
	rec = s.records[id]
	if rec == nil || rec.Status != "stopped" {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, errors.New("记录状态已变更"))
		return
	}
	now := time.Now().UTC()
	rec.RuntimeProfile = target
	rec.Status = "starting"
	rec.Stage = "checking"
	rec.Message = "检查本地制品、镜像身份与匹配缓存后启动"
	rec.Progress = progressCheckStart
	rec.PhaseProgress = 0
	rec.Accessible = false
	rec.Error = ""
	rec.StartedAt = now
	rec.FinishedAt = time.Time{}
	rec.StopExisting = req.StopExisting
	rec.StopProtected = req.StopProtected
	rec.ProtectedProcessIDs = append([]string(nil), req.ProtectedProcessIDs...)
	rec.Logs = append(rec.Logs, DeploymentLog{Timestamp: now, Message: rec.Message})
	if err := s.flushLocked(); err != nil {
		s.mu.Unlock()
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	out := cloneRecord(rec)
	s.mu.Unlock()

	go s.runDeploymentWorker(id)
	writeJSON(w, http.StatusAccepted, out)
}

func (s *Server) handlePiAgentConfig(w http.ResponseWriter, r *http.Request, id, userID string) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}
	s.mu.RLock()
	rec, ok := s.records[id]
	if !ok || rec.UserID != userID {
		s.mu.RUnlock()
		writeError(w, http.StatusNotFound, errors.New("部署记录不存在"))
		return
	}
	cp := cloneRecord(rec)
	s.mu.RUnlock()
	if strings.TrimSpace(cp.APIURL) == "" {
		writeError(w, http.StatusConflict, errors.New("部署完成后才能生成 Pi Agent 配置"))
		return
	}
	if !cp.PiAgentConfigReady {
		if err := s.persistPiAgentConfig(cp); err == nil {
			cp.PiAgentConfigReady = true
			s.updateRecord(id, func(r *DeploymentRecord) {
				r.PiAgentConfigPath = piAgentConfigTarget
				r.PiAgentConfigReady = true
			})
		}
	}
	writeJSON(w, http.StatusOK, PiAgentConfigResponse{
		Path:      piAgentConfigTarget,
		Config:    BuildPiAgentConfig(cp),
		Generated: cp.PiAgentConfigReady,
	})
}

func (s *Server) runDeploymentWorker(id string) {
	doneCh := make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), deploymentTimeout)

	s.mu.Lock()
	rec, ok := s.records[id]
	if !ok || rec.Status == "canceled" {
		s.mu.Unlock()
		cancel()
		close(doneCh)
		return
	}
	recordVal := cloneRecord(rec)
	s.cancelFuncs[id] = cancel
	s.workerDone[id] = doneCh
	s.mu.Unlock()

	defer func() {
		cancel()
		s.mu.Lock()
		delete(s.cancelFuncs, id)
		delete(s.workerDone, id)
		s.mu.Unlock()
		close(doneCh)
	}()

	settings, err := NormalizeDeployRequest(deployRequest{
		ContextLength: &recordVal.ContextLength,
		ModelProvider: recordVal.ModelProvider,
		Parameters:    recordVal.Parameters,
	})
	if err != nil {
		s.failDeployment(id, err.Error())
		return
	}

	host, target, err := s.resolveHostDeviceAndTarget(ctx, recordVal.UserID, recordVal.SerialNumber)
	if err != nil {
		s.failDeployment(id, err.Error())
		return
	}
	if !host.Online || !validLANIP(host.IP) {
		s.failDeployment(id, "算力舱离线或未上报有效私网 IP")
		return
	}

	s.updateRecord(id, func(r *DeploymentRecord) {
		if r.Status == "queued" {
			r.Status = "deploying"
		}
		r.Stage = "checking"
		r.Message = "核对模型快照与运行环境状态"
		r.Progress = progressCheckStart
		r.RuntimeProfile = target
	})

	modelRoot, cacheDir, err := s.prepareTargetArtifacts(ctx, recordVal, target, settings)
	if err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return
		}
		s.failDeployment(id, "准备模型或运行环境失败: "+err.Error())
		return
	}

	rt := s.manifests.Runtimes.Targets[target]
	groups := s.manifests.GroupsForTarget(target)

	if err := s.registerServiceOwnership(ctx, recordVal.SerialNumber, recordVal.UserID); err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return
		}
		s.failDeployment(id, "登记模型服务归属失败: "+err.Error())
		return
	}

	// Re-check resources & switch conflicts strictly AFTER all preparation completes (§7.1).
	assessment, err := s.switchResourcesBeforeStart(ctx, recordVal, target, settings)
	if err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return
		}
		s.failDeployment(id, err.Error())
		return
	}

	compose, err := BuildInferenceCompose(target, rt, groups, modelRoot, cacheDir, settings, assessment.Budget)
	if err != nil {
		s.failDeployment(id, "生成服务编排失败: "+err.Error())
		return
	}

	if err := s.deployAndWaitHealthy(ctx, recordVal, compose); err != nil {
		if errors.Is(ctx.Err(), context.Canceled) {
			return
		}
		s.failDeployment(id, "模型服务启动失败: "+err.Error())
		return
	}

	apiURL := fmt.Sprintf("http://%s:%d/v1", host.IP, config.DefaultAPIPort)
	s.completeDeployment(id, apiURL, assessment)
}

func (s *Server) completeDeployment(id, apiURL string, assessment memoryAssessment) {
	s.mu.RLock()
	rec, ok := s.records[id]
	var staged DeploymentRecord
	if ok {
		staged = cloneRecord(rec)
		staged.APIURL = apiURL
	}
	s.mu.RUnlock()

	var piReady bool
	if ok {
		if err := s.persistPiAgentConfig(staged); err == nil {
			piReady = true
		}
	}

	s.updateRecord(id, func(r *DeploymentRecord) {
		now := time.Now().UTC()
		r.Status = "succeeded"
		r.Stage = "ready"
		r.Message = "部署成功，推理服务已就绪"
		r.Progress = 100
		r.PhaseProgress = 100
		r.Estimated = false
		r.APIURL = apiURL
		r.Accessible = true
		r.Error = ""
		r.PiAgentConfigPath = piAgentConfigTarget
		r.PiAgentConfigReady = piReady
		r.EstimatedMemoryBytes = assessment.Budget.RequiredBytes
		r.RemainingMemoryBytes = maxInt64(0, assessment.AvailableBytes-assessment.Budget.RequiredBytes)
		r.FinishedAt = now
		r.ElapsedSeconds = now.Sub(r.StartedAt).Seconds()
		r.Logs = append(r.Logs, DeploymentLog{Timestamp: now, Message: "部署成功: " + apiURL})
	})
}

func (s *Server) failDeployment(id, errMsg string) {
	cleanErr := SanitizePublicText(errMsg)
	s.updateRecord(id, func(r *DeploymentRecord) {
		if r.Status == "canceled" {
			return
		}
		now := time.Now().UTC()
		r.Status = "failed"
		r.Stage = "failed"
		r.Message = "部署失败"
		r.Error = cleanErr
		r.Accessible = false
		r.FinishedAt = now
		r.ElapsedSeconds = now.Sub(r.StartedAt).Seconds()
		r.Logs = append(r.Logs, DeploymentLog{Timestamp: now, Message: cleanErr})
	})
}

func (s *Server) updateRecord(id string, fn func(*DeploymentRecord)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.records[id]
	if !ok {
		return
	}
	fn(rec)
	if len(rec.Logs) > 200 {
		rec.Logs = rec.Logs[len(rec.Logs)-200:]
	}
	_ = s.flushLocked()
}

func (s *Server) reusableRecordLocked(userID, serial string) *DeploymentRecord {
	var best *DeploymentRecord
	for _, r := range s.records {
		if r.UserID != userID || r.SerialNumber != serial {
			continue
		}
		if r.Status != "succeeded" && r.Status != "stopped" {
			continue
		}
		if _, running := s.cancelFuncs[r.ID]; running {
			continue
		}
		if best == nil || r.StartedAt.After(best.StartedAt) {
			best = r
		}
	}
	return best
}

func (s *Server) resolveHostDeviceAndTarget(ctx context.Context, userID, serial string) (HostDevice, string, error) {
	hosts, err := s.fetchHostDevices(ctx, userID)
	if err != nil {
		return HostDevice{}, "", fmt.Errorf("查询算力舱列表失败: %w", err)
	}
	for _, h := range hosts {
		if h.SerialNumber == serial {
			target, err := ClassifyDeviceTarget(h)
			return h, target, err
		}
	}
	return HostDevice{}, "", fmt.Errorf("未找到已绑定的算力舱 %s", serial)
}

func (s *Server) loadState() error {
	raw, err := os.ReadFile(s.dataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	if len(strings.TrimSpace(string(raw))) == 0 {
		return nil
	}
	var st persistedState
	if err := json.Unmarshal(raw, &st); err != nil {
		return fmt.Errorf("读取部署状态失败: %w", err)
	}
	for id, pr := range st.Records {
		if pr == nil {
			continue
		}
		rec := pr.DeploymentRecord
		rec.UserID = pr.UserID
		rec.StopExisting = pr.StopExisting
		rec.StopProtected = pr.StopProtected
		rec.ProtectedProcessIDs = append([]string(nil), pr.ProtectedProcessIDs...)
		if isRecordActive(rec.Status) {
			rec.Status = "failed"
			rec.Stage = "failed"
			rec.Message = "控制器重启，活动部署已中断，可点击重试恢复"
			rec.Error = "控制器重启导致进行中的部署中断，可点击重试继续复用已有进度"
		}
		s.records[id] = &rec
	}
	for k, v := range st.PreservedData {
		s.preservedData[k] = v
	}
	return nil
}

func (s *Server) flushLocked() error {
	st := persistedState{
		Records:       make(map[string]*persistedRecord, len(s.records)),
		PreservedData: s.preservedData,
	}
	for id, rec := range s.records {
		cp := cloneRecord(rec)
		st.Records[id] = &persistedRecord{
			DeploymentRecord:    cp,
			UserID:              rec.UserID,
			StopExisting:        rec.StopExisting,
			StopProtected:       rec.StopProtected,
			ProtectedProcessIDs: append([]string(nil), rec.ProtectedProcessIDs...),
		}
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.dataPath + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, s.dataPath)
}

func cloneRecord(r *DeploymentRecord) DeploymentRecord {
	cp := *r
	cp.Logs = append([]DeploymentLog(nil), r.Logs...)
	cp.Files = append([]ArtifactFileProgress(nil), r.Files...)
	cp.ProtectedProcessIDs = append([]string(nil), r.ProtectedProcessIDs...)
	if r.Parameters != nil {
		cp.Parameters = make(map[string]any, len(r.Parameters))
		for k, v := range r.Parameters {
			cp.Parameters[k] = v
		}
	}
	return cp
}

func isRecordActive(status string) bool {
	return status == "queued" || status == "deploying" || status == "starting"
}

func normalizeSerialList(req deployRequest) ([]string, error) {
	vals := req.SerialNumbers
	if len(vals) == 0 && strings.TrimSpace(req.SerialNumber) != "" {
		vals = []string{req.SerialNumber}
	}
	if len(vals) == 0 {
		return nil, errors.New("serialNumber 或 serialNumbers 不能为空")
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(vals))
	for _, v := range vals {
		s := strings.TrimSpace(v)
		if s == "" {
			return nil, errors.New("serialNumbers 不能包含空项")
		}
		if seen[s] {
			return nil, fmt.Errorf("serialNumber %q 重复", s)
		}
		seen[s] = true
		out = append(out, s)
	}
	return out, nil
}

func preservedKey(userID, serial string) string {
	return userID + ":" + serial
}

func requestUserID(r *http.Request) string {
	if uid := strings.TrimSpace(r.Header.Get("X-HC-User-ID")); uid != "" {
		return uid
	}
	return strings.TrimSpace(os.Getenv("LAZYCAT_USER_ID"))
}

func newRandomID() (string, error) {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(b[:]), nil
}

func decodeJSON(r *http.Request, target any) error {
	dec := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	dec.DisallowUnknownFields()
	return dec.Decode(target)
}

func writeJSON(w http.ResponseWriter, status int, val any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(val)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": SanitizePublicText(err.Error())})
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
