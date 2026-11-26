package lpk

import (
	"encoding/json"
	"strings"
	"time"
)

const (
	ProtocolVersion = "1.1"
	MinHostProtocol = "1.0"
)

type Descriptor struct {
	ProtocolVersion        string                   `json:"protocolVersion"`
	MinHostProtocolVersion string                   `json:"minHostProtocolVersion"`
	Package                string                   `json:"package"`
	Name                   string                   `json:"name"`
	Model                  ModelDescriptor          `json:"model"`
	Parameters             []ParameterDescriptor    `json:"parameters"`
	SupportedDevices       []string                 `json:"supportedDevices"`
	Capabilities           map[string]bool          `json:"capabilities"`
	Artifacts              map[string]ArtifactGroup `json:"artifacts"`
	RuntimeImage           RuntimeImageDescriptor   `json:"runtimeImage"`
	RuntimeImages          map[string]RuntimeImageDescriptor `json:"runtimeImages,omitempty"`
	RuntimeCache           *RuntimeCacheDescriptor  `json:"runtimeCache,omitempty"`
	ModelSources           *ModelSourcesDescriptor  `json:"modelSources,omitempty"`
	Modes                  []ModeDescriptor         `json:"modes"`
}

type ModelDescriptor struct {
	Name               string `json:"name"`
	TaskType           string `json:"taskType,omitempty"`
	InputMode          string `json:"inputMode,omitempty"`
	OutputType         string `json:"outputType,omitempty"`
	ContextLength      int    `json:"contextLength"`
	MinContextLength   int    `json:"minContextLength,omitempty"`
	MaxContextLength   int    `json:"maxContextLength,omitempty"`
	SafeReserveBytes   int64  `json:"safeReserveBytes,omitempty"`
	MaxConcurrency     int    `json:"maxConcurrency,omitempty"`
	NetworkDrivePath   string `json:"networkDrivePath"`
	ReuseAcrossDevices bool   `json:"reuseAcrossDevices"`
}

type ParameterDescriptor struct {
	ID          string                      `json:"id"`
	Label       string                      `json:"label"`
	Type        string                      `json:"type"`
	Unit        string                      `json:"unit,omitempty"`
	Default     any                         `json:"default,omitempty"`
	Min         any                         `json:"min,omitempty"`
	Max         any                         `json:"max,omitempty"`
	Step        any                         `json:"step,omitempty"`
	Description string                      `json:"description,omitempty"`
	Fixed       bool                        `json:"fixed,omitempty"`
	Options     []ParameterOptionDescriptor `json:"options,omitempty"`
}

type ParameterOptionDescriptor struct {
	Value string `json:"value"`
	Label string `json:"label"`
}

type RuntimeImageDescriptor struct {
	Source             string                    `json:"source"`
	Name               string                    `json:"name"`
	Image              string                    `json:"image"`
	ImageID            string                    `json:"imageId,omitempty"`
	ConfigImageID      string                    `json:"configImageId,omitempty"`
	RuntimeImageSHA256 string                    `json:"runtimeImageSha256,omitempty"`
	URL                string                    `json:"url,omitempty"`
	Providers          map[string]ProviderSource `json:"providers,omitempty"`
	Size               int64                     `json:"size"`
	SHA256             string                    `json:"sha256"`
}

type ProviderSource struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
	Path       string `json:"path,omitempty"`
}

type RuntimeCacheDescriptor struct {
	Name               string `json:"name"`
	URL                string `json:"url"`
	Size               int64  `json:"size"`
	SHA256             string `json:"sha256"`
	Fingerprint        string `json:"fingerprint"`
	RuntimeImageSHA256 string `json:"runtimeImageSha256"`
	LayoutVersion      int    `json:"layoutVersion"`
}

type ModelSourcesDescriptor struct {
	Policy          string                `json:"policy"`
	Detection       string                `json:"detection"`
	RegionProviders map[string]string     `json:"regionProviders"`
	Integrity       string                `json:"integrity"`
	Cache           string                `json:"cache"`
	TotalSize       int64                 `json:"totalSize"`
	Models          []ModelSourceEntry    `json:"models"`
}

type ModelSourceEntry struct {
	Role        string                    `json:"role"`
	Name        string                    `json:"name"`
	Destination string                    `json:"destination"`
	FileCount   int                       `json:"fileCount"`
	Size        int64                     `json:"size"`
	Providers   map[string]ProviderSource `json:"providers"`
}

type ArtifactGroup struct {
	Name       string         `json:"name"`
	Repository string         `json:"repository"`
	Files      []ArtifactFile `json:"files"`
}

type ArtifactFile struct {
	Name   string   `json:"name"`
	Size   int64    `json:"size"`
	SHA256 string   `json:"sha256"`
	URLs   []string `json:"urls,omitempty"`
	URL    string   `json:"url,omitempty"`
}

type ModeDescriptor struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	MaxNumSeqs  int      `json:"maxNumSeqs"`
	Description string   `json:"description"`
	Pros        []string `json:"pros"`
	Cons        []string `json:"cons"`
	Image       string   `json:"image"`
}

type HostDevicePlatformObject struct {
	Target         string `json:"target,omitempty"`
	Arch           string `json:"arch,omitempty"`
	JetPackVersion string `json:"jetpackVersion,omitempty"`
	JetPackMajor   int    `json:"jetpackMajor,omitempty"`
	JetPackMinor   int    `json:"jetpackMinor,omitempty"`
}

type HostDevice struct {
	SerialNumber string `json:"serialNumber"`
	Name         string `json:"name"`
	Model        string `json:"model"`
	L4TVersion   string `json:"l4tVersion,omitempty"`
	GoArch       string `json:"goArch,omitempty"`
	Platform     string `json:"platform,omitempty"`
	IP           string `json:"ip"`
	AgentPort    int    `json:"agentPort,omitempty"`
	Online       bool   `json:"online"`
	LastError    string `json:"lastError,omitempty"`
}

func (h *HostDevice) UnmarshalJSON(data []byte) error {
	type hostDeviceAlias HostDevice
	var aux struct {
		hostDeviceAlias
		RawPlatform json.RawMessage `json:"platform,omitempty"`
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*h = HostDevice(aux.hostDeviceAlias)
	h.Platform = parseFlexiblePlatform(aux.RawPlatform)
	return nil
}

func parseFlexiblePlatform(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	if strings.HasPrefix(trimmed, "\"") {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil {
			return strings.TrimSpace(s)
		}
		return ""
	}
	var obj HostDevicePlatformObject
	if err := json.Unmarshal(raw, &obj); err == nil {
		if t := strings.TrimSpace(obj.Target); t != "" {
			return t
		}
		parts := make([]string, 0, 2)
		if a := strings.TrimSpace(obj.Arch); a != "" {
			parts = append(parts, a)
		}
		if jp := strings.TrimSpace(obj.JetPackVersion); jp != "" {
			parts = append(parts, "jp"+jp)
		}
		return strings.Join(parts, "-")
	}
	return ""
}

type Device struct {
	SerialNumber string `json:"serialNumber"`
	Name         string `json:"name"`
	Model        string `json:"model"`
	L4TVersion   string `json:"l4tVersion,omitempty"`
	Platform     string `json:"platform,omitempty"`
	Target       string `json:"target,omitempty"`
	IP           string `json:"ip"`
	Port         int    `json:"port"`
	AgentPort    int    `json:"agentPort,omitempty"`
	Online       bool   `json:"online"`
	ModelReady   bool   `json:"modelReady"`
	LastError    string `json:"lastError,omitempty"`
}

func (d Device) NameOrSerial() string {
	if d.Name != "" {
		return d.Name
	}
	return d.SerialNumber
}

type HostDeviceResources struct {
	MemoryTotalBytes     int64 `json:"memoryTotalBytes"`
	MemoryUsedBytes      int64 `json:"memoryUsedBytes"`
	MemoryAvailableBytes int64 `json:"memoryAvailableBytes"`
	VRAMTotalBytes       int64 `json:"vramTotalBytes"`
	VRAMUsedBytes        int64 `json:"vramUsedBytes"`
	VRAMAvailableBytes   int64 `json:"vramAvailableBytes"`
	DiskAvailableBytes   int64 `json:"diskAvailableBytes"`
	DiskAvailableKnown   bool  `json:"diskAvailableKnown"`
	CapturedAt           int64 `json:"capturedAt"`
}

type HostCapabilities struct {
	ProtocolVersion    string          `json:"protocolVersion"`
	ArtifactDownloader string          `json:"artifactDownloader,omitempty"`
	Capabilities       map[string]bool `json:"capabilities,omitempty"`
}

type ArtifactStatusRequest struct {
	Namespace    string                    `json:"namespace"`
	Artifacts    map[string][]ArtifactFile `json:"artifacts"`
	VerifySHA256 bool                      `json:"verifySha256"`
}

type ArtifactStatus struct {
	Root      string          `json:"root"`
	Ready     bool            `json:"ready"`
	Artifacts map[string]bool `json:"artifacts,omitempty"`
	Missing   []string        `json:"missing,omitempty"`
}

type ArtifactPrepareRequest struct {
	Namespace    string                    `json:"namespace"`
	StoragePath  string                    `json:"storagePath"`
	Artifacts    map[string][]ArtifactFile `json:"artifacts"`
	VerifySHA256 bool                      `json:"verifySha256"`
}

type ArtifactPrepareResponse struct {
	JobID               string                 `json:"jobId"`
	Status              string                 `json:"status"`
	Phase               string                 `json:"phase,omitempty"`
	Root                string                 `json:"root"`
	Progress            int                    `json:"progress"`
	CurrentBytes        int64                  `json:"currentBytes,omitempty"`
	TotalBytes          int64                  `json:"totalBytes,omitempty"`
	SpeedBytesPerSecond int64                  `json:"speedBytesPerSecond,omitempty"`
	ETASeconds          int64                  `json:"etaSeconds,omitempty"`
	Estimated           bool                   `json:"estimated,omitempty"`
	TotalFiles          int                    `json:"totalFiles,omitempty"`
	CompletedFiles      int                    `json:"completedFiles,omitempty"`
	CurrentFile         string                 `json:"currentFile,omitempty"`
	CurrentFileIndex    int                    `json:"currentFileIndex,omitempty"`
	Files               []ArtifactFileProgress `json:"files,omitempty"`
	Message             string                 `json:"message,omitempty"`
	Error               string                 `json:"error,omitempty"`
	Aria2Log            string                 `json:"aria2Log,omitempty"`
}

type ImageStatusRequest struct {
	Image   string `json:"image"`
	ImageID string `json:"imageId,omitempty"`
}

type ImageStatusResponse struct {
	Image   string `json:"image"`
	ImageID string `json:"imageId,omitempty"`
	Ready   bool   `json:"ready"`
	Reason  string `json:"reason,omitempty"`
}

type ImageLoadRequest struct {
	StoragePath       string `json:"storagePath"`
	Image             string `json:"image,omitempty"`
	ImageID           string `json:"imageId,omitempty"`
	Name              string `json:"name"`
	Size              int64  `json:"size"`
	SHA256            string `json:"sha256"`
	RequiredDiskBytes int64  `json:"requiredDiskBytes,omitempty"`
	Async             bool   `json:"async,omitempty"`
}

type ImageLoadResponse struct {
	JobID               string `json:"jobId,omitempty"`
	Status              string `json:"status,omitempty"`
	Phase               string `json:"phase,omitempty"`
	Message             string `json:"message,omitempty"`
	Name                string `json:"name,omitempty"`
	Size                int64  `json:"size,omitempty"`
	Loaded              bool   `json:"loaded,omitempty"`
	Reused              bool   `json:"reused,omitempty"`
	Progress            int    `json:"progress"`
	CurrentBytes        int64  `json:"currentBytes,omitempty"`
	TotalBytes          int64  `json:"totalBytes,omitempty"`
	SpeedBytesPerSecond int64  `json:"speedBytesPerSecond,omitempty"`
	ETASeconds          int64  `json:"etaSeconds,omitempty"`
	Estimated           bool   `json:"estimated,omitempty"`
	Error               string `json:"error,omitempty"`
}

type CacheRestoreRequest struct {
	StoragePath        string `json:"storagePath"`
	ServiceID          string `json:"serviceId"`
	Name               string `json:"name"`
	Size               int64  `json:"size"`
	SHA256             string `json:"sha256"`
	Fingerprint        string `json:"fingerprint"`
	RuntimeImageSHA256 string `json:"runtimeImageSha256"`
}

type CacheRestoreResponse struct {
	Name     string `json:"name"`
	Size     int64  `json:"size"`
	Restored bool   `json:"restored"`
	Reused   bool   `json:"reused,omitempty"`
}

type HostProcessStatus struct {
	OllamaPs []HostOllamaProcess          `json:"ollamaPs"`
	Services map[string]HostServiceStatus `json:"services"`
}

type HostOllamaProcess struct {
	PackageID   string `json:"pkgId"`
	Name        string `json:"name"`
	Model       string `json:"model"`
	MemoryUsage uint64 `json:"memoryUsage"`
}

type HostServiceStatus struct {
	MemoryUsage uint64 `json:"memoryUsage"`
}

type ProcessStopRequest struct {
	PackageIDs []string `json:"packageIds"`
}

type HostServiceResourceUsage struct {
	DiskUsageBytes   int64 `json:"diskUsageBytes"`
	ReclaimableBytes int64 `json:"reclaimableBytes"`
	Estimated        bool  `json:"estimated"`
}

type ServiceOwnershipRequest struct {
	PackageID string `json:"packageId"`
	ServiceID string `json:"serviceId"`
}

type ServiceOwnershipResponse struct {
	Registered bool `json:"registered"`
}

type DeployServiceRequest struct {
	ServiceID      string            `json:"serviceId"`
	OwnerPackageID string            `json:"ownerPackageId,omitempty"`
	Compose        string            `json:"compose"`
	ForceRecreate  bool              `json:"forceRecreate"`
	DirOverWrite   bool              `json:"dirOverWrite"`
	ExtraEnv       map[string]string `json:"extraEnv,omitempty"`
}

type HostServiceDeployStatus struct {
	ServiceID         string `json:"serviceId"`
	Stage             string `json:"stage"`
	Status            string `json:"status,omitempty"`
	Progress          int    `json:"progress"`
	AggregateProgress bool   `json:"aggregateProgress,omitempty"`
	PhaseProgress     int    `json:"phaseProgress,omitempty"`
	Estimated         bool   `json:"estimated,omitempty"`
	Message           string `json:"message,omitempty"`
	State             string `json:"state,omitempty"`
	Health            string `json:"health,omitempty"`
	CurrentImage      string `json:"currentImage,omitempty"`
	CurrentLayer      string `json:"currentLayer,omitempty"`
	Current           int64  `json:"current,omitempty"`
	Total             int64  `json:"total,omitempty"`
	StartedAt         int64  `json:"startedAt,omitempty"`
	UpdatedAt         int64  `json:"updatedAt,omitempty"`
	FinishedAt        int64  `json:"finishedAt,omitempty"`
	Error             string `json:"error,omitempty"`
}

type HealthRequest struct {
	Port int    `json:"port"`
	Path string `json:"path"`
}

type HealthResponse struct {
	Healthy    bool   `json:"healthy"`
	StatusCode int    `json:"statusCode,omitempty"`
	Body       string `json:"body,omitempty"`
	Error      string `json:"error,omitempty"`
}

type ArtifactFileProgress struct {
	Role            string `json:"role"`
	Name            string `json:"name"`
	Status          string `json:"status"`
	DownloadedBytes int64  `json:"downloadedBytes"`
	TotalBytes      int64  `json:"totalBytes"`
	Progress        int    `json:"progress"`
}

type ArtifactProgress struct {
	TotalFiles          int                    `json:"totalFiles,omitempty"`
	CompletedFiles      int                    `json:"completedFiles,omitempty"`
	CurrentFile         string                 `json:"currentFile,omitempty"`
	CurrentFileIndex    int                    `json:"currentFileIndex,omitempty"`
	Progress            int                    `json:"progress"`
	Phase               string                 `json:"phase,omitempty"`
	CurrentBytes        int64                  `json:"currentBytes,omitempty"`
	TotalBytes          int64                  `json:"totalBytes,omitempty"`
	SpeedBytesPerSecond int64                  `json:"speedBytesPerSecond,omitempty"`
	ETASeconds          int64                  `json:"etaSeconds,omitempty"`
	Estimated           bool                   `json:"estimated,omitempty"`
	Message             string                 `json:"message,omitempty"`
	Files               []ArtifactFileProgress `json:"files,omitempty"`
	ImageLoaded         bool                   `json:"-"`
}

type DeploymentLog struct {
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message"`
}

type DeploymentRecord struct {
	ID                   string                 `json:"id"`
	ModelName            string                 `json:"modelName"`
	ModelVersion         string                 `json:"modelVersion"`
	ModelID              string                 `json:"modelId"`
	APIKey               string                 `json:"apiKey"`
	SerialNumber         string                 `json:"serialNumber"`
	DeviceName           string                 `json:"deviceName"`
	DeviceModel          string                 `json:"deviceModel"`
	RuntimeProfile       string                 `json:"runtimeProfile,omitempty"`
	Mode                 string                 `json:"mode"`
	MaxNumSeqs           int                    `json:"maxNumSeqs"`
	Concurrency          int                    `json:"concurrency"`
	ContextLength        int                    `json:"contextLength"`
	ModelProvider        string                 `json:"modelProvider"`
	Parameters           map[string]any         `json:"parameters,omitempty"`
	EstimatedMemoryBytes int64                  `json:"estimatedMemoryBytes,omitempty"`
	RemainingMemoryBytes int64                  `json:"remainingMemoryBytes,omitempty"`
	MemoryUsageBytes     uint64                 `json:"memoryUsageBytes"`
	DiskUsageBytes       int64                  `json:"diskUsageBytes"`
	ReclaimableBytes     int64                  `json:"reclaimableBytes"`
	DiskUsageEstimated   bool                   `json:"diskUsageEstimated,omitempty"`
	MaxConcurrency       int                    `json:"maxConcurrency,omitempty"`
	Image                string                 `json:"image"`
	Status               string                 `json:"status"`
	Stage                string                 `json:"stage"`
	Message              string                 `json:"message"`
	StartedAt            time.Time              `json:"startedAt"`
	FinishedAt           time.Time              `json:"finishedAt,omitempty"`
	ElapsedSeconds       float64                `json:"elapsedSeconds"`
	Progress             int                    `json:"progress"`
	PhaseProgress        int                    `json:"phaseProgress"`
	CurrentBytes         int64                  `json:"currentBytes,omitempty"`
	TotalBytes           int64                  `json:"totalBytes,omitempty"`
	SpeedBytesPerSecond  int64                  `json:"speedBytesPerSecond,omitempty"`
	ETASeconds           int64                  `json:"etaSeconds,omitempty"`
	Estimated            bool                   `json:"estimated,omitempty"`
	TotalFiles           int                    `json:"totalFiles,omitempty"`
	CompletedFiles       int                    `json:"completedFiles,omitempty"`
	CurrentFile          string                 `json:"currentFile,omitempty"`
	CurrentFileIndex     int                    `json:"currentFileIndex,omitempty"`
	Files                []ArtifactFileProgress `json:"files,omitempty"`
	APIURL               string                 `json:"apiUrl,omitempty"`
	Accessible           bool                   `json:"accessible"`
	Error                string                 `json:"error,omitempty"`
	PiAgentConfigPath    string                 `json:"piAgentConfigPath,omitempty"`
	PiAgentConfigReady   bool                   `json:"piAgentConfigReady,omitempty"`
	Logs                 []DeploymentLog        `json:"logs,omitempty"`

	// Internal tracking fields (never leaked externally).
	StopExisting        bool     `json:"-"`
	StopProtected       bool     `json:"-"`
	ProtectedProcessIDs []string `json:"-"`
	UserID              string   `json:"-"`
}

type PiAgentConfigResponse struct {
	Path      string         `json:"path"`
	Config    map[string]any `json:"config"`
	Generated bool           `json:"generated"`
}
