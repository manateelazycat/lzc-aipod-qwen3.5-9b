package lpk

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"qwen35-9b-deployer/internal/config"
)

func TestDevicesEndpointDecodesObjectPlatform(t *testing.T) {
	const hostFixture = `[
		{"serialNumber":"1420225016421","name":"orin-jetpack6.2.1","model":"NVIDIA Jetson AGX Orin","l4tVersion":"36.4.3","goArch":"arm64","platform":{"arch":"agxorin","jetpackVersion":"6.2.1","jetpackMajor":6,"jetpackMinor":2,"target":"agxorin-jp6"},"ip":"192.168.1.224","agentPort":40793,"online":true},
		{"serialNumber":"1421826028551","name":"thor-t5000-1","model":"NVIDIA Jetson AGX Thor","l4tVersion":"39.2.0","goArch":"arm64","platform":{"arch":"thor","jetpackVersion":"7.2","jetpackMajor":7,"jetpackMinor":2,"target":"thor-jp7"},"ip":"192.168.1.127","agentPort":40793,"online":true},
		{"serialNumber":"1422326009786","name":"thor-t5000-2","model":"NVIDIA Jetson AGX Thor","l4tVersion":"39.2.0","goArch":"arm64","platform":{"arch":"thor","jetpackVersion":"7.2","jetpackMajor":7,"jetpackMinor":2,"target":"thor-jp7"},"ip":"192.168.1.141","agentPort":40793,"online":true},
		{"serialNumber":"1422624327649","name":"orin-jetpack7.2.1","model":"NVIDIA Jetson AGX Orin","l4tVersion":"39.2.1","goArch":"arm64","platform":{"arch":"agxorin","jetpackVersion":"7.2.1","jetpackMajor":7,"jetpackMinor":2,"target":"agxorin-jp7"},"ip":"192.168.1.176","agentPort":40793,"online":true},
		{"serialNumber":"1423025073160","name":"thor-dev","model":"NVIDIA Jetson AGX Thor","l4tVersion":"39.2.0","goArch":"arm64","platform":{"arch":"thor","jetpackVersion":"7.2","jetpackMajor":7,"jetpackMinor":2,"target":"thor-jp7"},"ip":"192.168.1.122","agentPort":40793,"online":true},
		{"serialNumber":"1424625060535","name":"lzc-aipod-ghhlvl","model":"NVIDIA Jetson AGX Thor","l4tVersion":"39.2.1","goArch":"arm64","platform":{"arch":"thor","jetpackVersion":"7.2.1","jetpackMajor":7,"jetpackMinor":2,"target":"thor-jp7"},"ip":"192.168.1.112","agentPort":40793,"online":true}
	]`

	mockHost := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model-host/v1/devices" {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(hostFixture))
			return
		}
		http.NotFound(w, r)
	}))
	defer mockHost.Close()

	srv, err := NewServer(mockHost.URL, t.TempDir())
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	defer srv.Close()

	req := httptest.NewRequest(http.MethodGet, "/api/v1/devices", nil)
	req.Header.Set("X-HC-User-ID", "admin")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("GET /api/v1/devices returned %d: %s", rec.Code, rec.Body.String())
	}

	var got []Device
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode /api/v1/devices response: %v", err)
	}
	wantTargets := map[string]string{
		"1420225016421": "",
		"1421826028551": config.TargetThor,
		"1422326009786": config.TargetThor,
		"1422624327649": config.TargetOrin,
		"1423025073160": config.TargetThor,
		"1424625060535": config.TargetThor,
	}
	if len(got) != len(wantTargets) {
		t.Fatalf("got %d devices, want %d (4 Thor + 1 Orin JP7 + 1 Orin JP6 with L4T error)", len(got), len(wantTargets))
	}
	for _, dev := range got {
		want, ok := wantTargets[dev.SerialNumber]
		if !ok {
			t.Fatalf("unexpected device serial %q", dev.SerialNumber)
		}
		if dev.Target != want {
			t.Fatalf("device %s (%s) target=%q, want %q", dev.SerialNumber, dev.Name, dev.Target, want)
		}
		if want == "" && dev.LastError == "" {
			t.Fatalf("device %s (%s) expected non-empty LastError for unsupported L4T", dev.SerialNumber, dev.Name)
		}
		if dev.Platform == "" {
			t.Fatalf("device %s expected non-empty decoded platform", dev.SerialNumber)
		}
	}
}

func TestServiceOwnershipRegistrationAndConflictBeforeStop(t *testing.T) {
	var (
		gotPath string
		gotBody ServiceOwnershipRequest
		mode    string
	)
	mockHost := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model-host/v1/devices/sn-123/services/ownership" {
			gotPath = r.URL.Path
			_ = json.NewDecoder(r.Body).Decode(&gotBody)
			switch mode {
			case "ok":
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"registered":true}`))
			case "unregistered":
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"registered":false}`))
			case "conflict":
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusConflict)
				_, _ = w.Write([]byte(`{"error":"service belongs to another application or user"}`))
			default:
				http.NotFound(w, r)
			}
			return
		}
		http.NotFound(w, r)
	}))
	defer mockHost.Close()

	srv, err := NewServer(mockHost.URL, t.TempDir())
	if err != nil {
		t.Fatalf("NewServer failed: %v", err)
	}
	defer srv.Close()

	ctx := context.Background()
	mode = "ok"
	if err := srv.registerServiceOwnership(ctx, "sn-123", "admin"); err != nil {
		t.Fatalf("expected registered=true to pass, got %v", err)
	}
	if gotPath != "/model-host/v1/devices/sn-123/services/ownership" {
		t.Fatalf("unexpected path %q", gotPath)
	}
	if gotBody.PackageID != config.PackageID || gotBody.ServiceID != config.DefaultServiceName {
		t.Fatalf("unexpected body %+v", gotBody)
	}

	mode = "legacy404"
	if err := srv.registerServiceOwnership(ctx, "sn-123", "admin"); err != nil {
		t.Fatalf("expected 404 legacy fallback to pass, got %v", err)
	}

	mode = "unregistered"
	if err := srv.registerServiceOwnership(ctx, "sn-123", "admin"); err == nil {
		t.Fatal("expected registered=false to fail")
	}

	mode = "conflict"
	if err := srv.registerServiceOwnership(ctx, "sn-123", "admin"); err == nil {
		t.Fatal("expected 409 conflict to fail")
	}
}
