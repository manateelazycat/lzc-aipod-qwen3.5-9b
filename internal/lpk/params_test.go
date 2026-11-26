package lpk

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	"qwen35-9b-deployer/internal/config"
)

func TestParameterContractEnforcement(t *testing.T) {
	srv := newTestServer(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, []any{})
	}))

	// Valid requirements queries
	for _, query := range []string{
		"/api/v1/deployments/requirements?serialNumber=sn-1",
		"/api/v1/deployments/requirements?serialNumber=sn-1&contextLength=131072&modelProvider=modelscope",
		"/api/v1/deployments/requirements?serialNumber=sn-1&contextLength=32768&modelProvider=huggingface",
	} {
		req := httptest.NewRequest(http.MethodGet, query, nil)
		req.Header.Set("X-HC-User-ID", "u1")
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200 for %s, got %d (%s)", query, rec.Code, rec.Body.String())
		}
	}

	// Invalid requirements queries (must return 400)
	for _, query := range []string{
		"/api/v1/deployments/requirements?serialNumber=sn-1&contextLength=8192",
		"/api/v1/deployments/requirements?serialNumber=sn-1&contextLength=131073",
		"/api/v1/deployments/requirements?serialNumber=sn-1&contextLength=262144&maxModelLen=131072",
		"/api/v1/deployments/requirements?serialNumber=sn-1&modelProvider=cdn",
		"/api/v1/deployments/requirements?serialNumber=sn-1&concurrency=2",
		"/api/v1/deployments/requirements?serialNumber=sn-1&unknownParam=1",
	} {
		req := httptest.NewRequest(http.MethodGet, query, nil)
		req.Header.Set("X-HC-User-ID", "u1")
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for %s, got %d (%s)", query, rec.Code, rec.Body.String())
		}
	}

	// Invalid POST /deployments parameter payloads
	badPayloads := []string{
		`{"serialNumber":"sn-1","contextLength":131072,"parameters":{"contextLength":65536}}`,
		`{"serialNumber":"sn-1","parameters":{"unknownKey":"abc"}}`,
		`{"serialNumber":"sn-1","concurrency":1}`,
		`{"serialNumber":"sn-1","parameters":{"contextLength":131072.5}}`,
		`{"serialNumber":"sn-1","parameters":{"contextLength":524288}}`,
		`{"serialNumber":"sn-1","parameters":{"modelProvider":"registry"}}`,
		`{"serialNumber":"sn-1","mode":"turbo"}`,
	}
	for _, body := range badPayloads {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/deployments", bytes.NewBufferString(body))
		req.Header.Set("X-HC-User-ID", "u1")
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for payload %s, got %d (%s)", body, rec.Code, rec.Body.String())
		}
	}

	// Valid settings check
	okCtx := config.MaxContextLength
	s, err := NormalizeDeployRequest(deployRequest{
		ContextLength: &okCtx,
		ModelProvider: "huggingface",
		Parameters: map[string]any{
			"contextLength": okCtx,
			"modelProvider": "huggingface",
		},
	})
	if err != nil || s.ContextLength != config.MaxContextLength || s.ModelProvider != "huggingface" || s.Concurrency != config.MaxConcurrency {
		t.Fatalf("unexpected normalized settings: %+v, err=%v", s, err)
	}
}
