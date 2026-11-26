package lpk

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func buildTestTarGz(t *testing.T, entries []struct {
	name string
	mode int64
	body []byte
	typ  byte
	link string
}) ([]byte, string) {
	t.Helper()
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)
	for _, e := range entries {
		typ := e.typ
		if typ == 0 {
			typ = tar.TypeReg
		}
		hdr := &tar.Header{
			Name:     e.name,
			Mode:     e.mode,
			Size:     int64(len(e.body)),
			Typeflag: typ,
			Linkname: e.link,
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("WriteHeader(%s): %v", e.name, err)
		}
		if len(e.body) > 0 {
			if _, err := tw.Write(e.body); err != nil {
				t.Fatalf("Write(%s): %v", e.name, err)
			}
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	raw := buf.Bytes()
	sum := sha256.Sum256(raw)
	return raw, hex.EncodeToString(sum[:])
}

func runBootstrap(t *testing.T, manifestPath, dataDir, devArchive string) (string, string, error) {
	t.Helper()
	scriptPath, err := filepath.Abs(filepath.Join("..", "..", "lpk", "content", "bootstrap.sh"))
	if err != nil {
		t.Fatalf("abs bootstrap path: %v", err)
	}
	cmd := exec.Command("/bin/sh", scriptPath)
	cmd.Env = []string{
		"PATH=" + os.Getenv("PATH"),
		"QWEN35_9B_CONTROL_MANIFEST=" + manifestPath,
		"QWEN35_9B_DATA_DIR=" + dataDir,
		"QWEN35_9B_DEV_CONTROL_ARCHIVE=" + devArchive,
	}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err = cmd.Run()
	return stdout.String(), stderr.String(), err
}

func writeControlManifest(t *testing.T, dir, binSHA, archiveSHA string, archiveSize int) string {
	t.Helper()
	doc := map[string]any{
		"schema":         "model-host.control-manifest.v1",
		"status":         "locked",
		"package":        "cloud.lazycat.aipod.qwen35-9b-defiant-fable",
		"version":        "0.1.0",
		"binaryName":     "qwen35-9b-lpk",
		"binarySha256":   binSHA,
		"size":           archiveSize,
		"sha256":         archiveSHA,
		"modelscopeUrl":  "",
		"huggingfaceUrl": "",
	}
	raw, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	path := filepath.Join(dir, "control-manifest.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

func TestBootstrapValidArchiveAndDevOverride(t *testing.T) {
	tmp := t.TempDir()
	binBody := []byte("#!/bin/sh\necho \"bootstrapped:${QWEN35_9B_CONTROLLER_SHA}\"\n")
	binSum := sha256.Sum256(binBody)
	binSHA := hex.EncodeToString(binSum[:])

	archiveBytes, archiveSHA := buildTestTarGz(t, []struct {
		name string
		mode int64
		body []byte
		typ  byte
		link string
	}{
		{name: "qwen35-9b-lpk", mode: 0o755, body: binBody},
	})
	archivePath := filepath.Join(tmp, "controller.tar.gz")
	if err := os.WriteFile(archivePath, archiveBytes, 0o600); err != nil {
		t.Fatalf("write archive: %v", err)
	}
	manifestPath := writeControlManifest(t, tmp, binSHA, archiveSHA, len(archiveBytes))
	dataDir := filepath.Join(tmp, "var")

	stdout, stderr, err := runBootstrap(t, manifestPath, dataDir, archivePath)
	if err != nil {
		t.Fatalf("bootstrap failed: %v\nstdout=%s\nstderr=%s", err, stdout, stderr)
	}
	want := "bootstrapped:" + archiveSHA
	if strings.TrimSpace(stdout) != want {
		t.Fatalf("unexpected stdout: got %q, want %q", strings.TrimSpace(stdout), want)
	}
	// Run a second time without dev archive to confirm cached unpack & verify_root path works.
	stdout2, stderr2, err2 := runBootstrap(t, manifestPath, dataDir, "")
	if err2 != nil {
		t.Fatalf("second cached bootstrap failed: %v\nstdout=%s\nstderr=%s", err2, stdout2, stderr2)
	}
	if strings.TrimSpace(stdout2) != want {
		t.Fatalf("unexpected second stdout: got %q, want %q", strings.TrimSpace(stdout2), want)
	}
}

func TestBootstrapRejectsPathTraversal(t *testing.T) {
	tmp := t.TempDir()
	binBody := []byte("#!/bin/sh\necho should-not-run\n")
	binSum := sha256.Sum256(binBody)
	binSHA := hex.EncodeToString(binSum[:])

	archiveBytes, archiveSHA := buildTestTarGz(t, []struct {
		name string
		mode int64
		body []byte
		typ  byte
		link string
	}{
		{name: "qwen35-9b-lpk", mode: 0o755, body: binBody},
		{name: "../escape.txt", mode: 0o644, body: []byte("bad")},
	})
	archivePath := filepath.Join(tmp, "traversal.tar.gz")
	if err := os.WriteFile(archivePath, archiveBytes, 0o600); err != nil {
		t.Fatalf("write archive: %v", err)
	}
	manifestPath := writeControlManifest(t, tmp, binSHA, archiveSHA, len(archiveBytes))
	dataDir := filepath.Join(tmp, "var")

	stdout, stderr, err := runBootstrap(t, manifestPath, dataDir, archivePath)
	if err == nil {
		t.Fatalf("expected path traversal rejection, got nil error (stdout=%s)", stdout)
	}
	if !strings.Contains(stderr, "Invalid controller archive path") {
		t.Fatalf("expected Invalid controller archive path in stderr, got %q", stderr)
	}
}

func TestBootstrapRejectsCorruptChecksum(t *testing.T) {
	tmp := t.TempDir()
	binBody := []byte("#!/bin/sh\necho ok\n")
	binSum := sha256.Sum256(binBody)
	binSHA := hex.EncodeToString(binSum[:])

	archiveBytes, archiveSHA := buildTestTarGz(t, []struct {
		name string
		mode int64
		body []byte
		typ  byte
		link string
	}{
		{name: "qwen35-9b-lpk", mode: 0o755, body: binBody},
	})
	// Corrupt one byte in the archive while manifest still demands original sha256.
	corrupt := append([]byte(nil), archiveBytes...)
	corrupt[len(corrupt)-1] ^= 0xff
	corruptPath := filepath.Join(tmp, "corrupt.tar.gz")
	if err := os.WriteFile(corruptPath, corrupt, 0o600); err != nil {
		t.Fatalf("write corrupt archive: %v", err)
	}
	manifestPath := writeControlManifest(t, tmp, binSHA, archiveSHA, len(archiveBytes))
	dataDir := filepath.Join(tmp, "var")

	stdout, stderr, err := runBootstrap(t, manifestPath, dataDir, corruptPath)
	if err == nil {
		t.Fatalf("expected checksum mismatch failure, got nil error (stdout=%s)", stdout)
	}
	if strings.Contains(stdout, "ok") {
		t.Fatalf("corrupt archive must never execute payload, stdout=%q stderr=%q", stdout, stderr)
	}
}
