package config

const (
	PackageID            = "cloud.lazycat.aipod.qwen35-9b-defiant-fable"
	Version              = "0.2.0"
	PublicServiceName    = "Qwen 3.5 9B"
	PublicModelID        = "qwen3.5-9b-defiant-fable"
	DefaultServiceName   = "qwen35-9b-defiant-fable"
	DefaultAPIPort       = 18156
	DefaultAPIKey        = "local"
	DefaultPort          = 8080
	DefaultBackend       = "http://host.lzcapp:40794"
	DeviceModelNamespace = "qwen35-9b-defiant-fable/models"
	ArtifactNamespace    = "cloud.lazycat.aipod.qwen35-9b-defiant-fable/models"
	NetworkDriveDisplay  = "/AI 模型/Qwen 3.5 9B Defiant Fable/"

	TargetThor = "thor"
	TargetOrin = "orin"

	EngineLlamaCpp = "llamacpp"

	MinContextLength     = 16384
	MaxContextLength     = 262144
	DefaultContextLength = 131072
	ContextLengthStep    = 1024
	MaxConcurrency       = 8
	DefaultModelProvider = "modelscope"

	// Upstream source identity pinned to immutable commit.
	UpstreamRepository = "DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF"
	UpstreamRevision   = "4d47954f2ef8277567a20508dd3f88def5404455"
	UpstreamLicense    = "Apache-2.0"
)
