package manifests

import _ "embed"

//go:embed models.json
var ModelsJSON []byte

//go:embed runtimes.json
var RuntimesJSON []byte

//go:embed caches.json
var CachesJSON []byte
