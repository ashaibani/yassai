// localmodeleval sanity-checks a fine-tuned MiniCPM5 GGUF through the SAME
// engine production uses (internal/localllm: spawned llama-server + gates).
// Content-level slips are visible but the key signal is framing: parseable
// run_python tool calls that execute. Run it on any new GGUF artefact before
// pointing CI's LOCAL_MODEL_URL at it.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ashaibani/yassai/internal/localllm"
)

func main() {
	libPath := flag.String("lib", firstNonEmpty(os.Getenv("YZMA_LIB"), "/opt/llama"), "directory with llama-server and its shared libraries")
	ftPath := flag.String("ft", "models/minicpm5/MiniCPM5-yassai-Q4_K_M.gguf", "fine-tuned GGUF model")
	threads := flag.Int("threads", 6, "decode threads")
	timeout := flag.Duration("timeout", 120*time.Second, "per-task budget")
	flag.Parse()

	solver, err := localllm.New(localllm.Config{
		ModelPath: *ftPath,
		LibPath:   *libPath,
		Threads:   *threads,
		Timeout:   *timeout,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
	defer solver.Close()
	fmt.Println("boot canary passed - engine decodes coherently")

	cases := []struct{ name, prompt, wantSub string }{
		{
			"warehouse",
			"A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
			"1672",
		},
		{
			"trains",
			"A train leaves City A at 08:00 travelling toward City B at 90 km/h. A second train leaves City B at 09:30 travelling toward City A at 110 km/h. The distance between the cities is 450 km. At what time do the trains meet, and how far from City A is the meeting point?",
			"11:04",
		},
		{
			"pets_logic",
			"Three friends - Maya, Noah, and Omar - each own a different pet: cat, dog, and parrot. Noah is allergic to fur. Maya does not own the parrot. Omar does not own the dog. State each person's pet.",
			"Maya: dog",
		},
	}
	passed := 0
	for _, c := range cases {
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		res := solver.SolveTask(ctx, c.prompt)
		cancel()
		switch {
		case res.OK && strings.Contains(res.Answer, c.wantSub):
			passed++
			fmt.Printf("PASS %-12s %s\n", c.name, res.Answer)
		case res.OK:
			fmt.Printf("SOFT %-12s accepted but unexpected: %s\n", c.name, res.Answer)
		default:
			fmt.Printf("FAIL %-12s %s\n", c.name, res.Reason)
			if res.Code != "" {
				fmt.Printf("     code: %.200s\n", res.Code)
			}
		}
	}
	fmt.Printf("score %d/%d\n", passed, len(cases))
	if passed == 0 {
		os.Exit(1)
	}
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
