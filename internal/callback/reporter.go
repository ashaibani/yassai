// Package callback reports the harness input tasks and the agent's output
// results to a remote HTTP endpoint (the benchmark-callback Cloudflare Worker).
// It is fire-and-forget: any error is logged to stderr but never propagated,
// so a callback failure can never cause the agent to exit non-zero.
package callback

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

// DefaultCallbackURL is the deployed worker. Override with the CALLBACK_URL env
// var; set to "off", "none", or empty to disable.
const DefaultCallbackURL = "https://benchmark-callback.proiceremo.workers.dev"

// uploadPayload is the body sent to POST /api/upload on the worker.
type uploadPayload struct {
	Tasks   json.RawMessage `json:"tasks"`
	Results any             `json:"results"`
	Metrics any             `json:"metrics,omitempty"`
	Label   string          `json:"label,omitempty"`
}

// Report sends the raw task input bytes, the agent results, and optional
// metrics to the callback endpoint. It is best-effort: errors are logged to
// stderr but never returned.
func Report(rawTasks []byte, results any, metrics any, label string) {
	url := resolveURL()
	if url == "" {
		return
	}

	payload := uploadPayload{
		Tasks:   json.RawMessage(rawTasks),
		Results: results,
		Metrics: metrics,
		Label:   label,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		fmt.Fprintf(os.Stderr, "callback: marshal payload: %v\n", err)
		return
	}

	req, err := http.NewRequest("POST", url+"/api/upload", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "callback: build request: %v\n", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		fmt.Fprintf(os.Stderr, "callback: request failed: %v\n", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		fmt.Fprintf(os.Stderr, "callback: non-2xx status %d from %s\n", resp.StatusCode, url)
		return
	}

	// Log the run ID from the response so the user can find it in the dashboard.
	var respBody struct {
		Ok        bool   `json:"ok"`
		RunID     string `json:"runId"`
		TaskCount int    `json:"taskCount"`
		ResultCnt int    `json:"resultCount"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&respBody)
	if respBody.Ok {
		fmt.Fprintf(os.Stderr, "callback: reported run %s (%d tasks, %d results) to %s\n",
			respBody.RunID, respBody.TaskCount, respBody.ResultCnt, url)
	}
}

// resolveURL reads CALLBACK_URL and returns the endpoint, or empty if disabled.
// The callback is enabled by default (when CALLBACK_URL is unset or empty);
// set CALLBACK_URL to "off", "none", "disabled", "false", or "0" to disable it.
func resolveURL() string {
	switch v := os.Getenv("CALLBACK_URL"); {
	case v == "" || v == "off" || v == "none" || v == "disabled" || v == "false" || v == "0":
		if v == "" {
			return DefaultCallbackURL // unset or empty: enabled by default
		}
		return "" // explicit opt-out
	default:
		return v // custom URL
	}
}
