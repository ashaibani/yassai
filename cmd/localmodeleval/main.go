package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/message"
	"github.com/hybridgroup/yzma/pkg/template"
)

// Must stay byte-identical to internal/localllm and the SFT data builder
// (JSON tool-call contract; the fenced variant regressed the 1B LoRA).
const systemPrompt = `You are yassai-local, a small local specialist for math and logic tasks.
Use run_python for arithmetic, percentages, schedules, projections, combinatorics, and constraint checks.
Never do these calculations in your head.
When a tool is needed, respond with exactly one tool call and no prose:
<tool_call>{"name":"run_python","arguments":{"code":"..."}}</tool_call>
The Python must compute from named variables and print concise final values.
After receiving a run_python result, return only the final answer requested by the user.`

type evalCase struct {
	Name             string
	Prompt           string
	ExpectedToolOut  string
	ExpectedFinal    string
	MaxTokens        int32
	RequireTool      bool
	RequireFinalOnly bool
}

type loadedModel struct {
	name         string
	path         string
	model        llama.Model
	ctx          llama.Context
	vocab        llama.Vocab
	chatTemplate string
}

func main() {
	libPath := flag.String("lib", firstNonEmpty(os.Getenv("YZMA_LIB"), "/opt/homebrew/lib"), "path containing llama.cpp shared libraries")
	basePath := flag.String("base", "models/minicpm5/MiniCPM5-1B-base-Q4_K_M.gguf", "base GGUF model")
	ftPath := flag.String("ft", "models/minicpm5/MiniCPM5-yassai-Q4_K_M.gguf", "fine-tuned GGUF model")
	maxTokens := flag.Int("max-tokens", 224, "maximum generated tokens per case")
	nctx := flag.Uint("ctx", 4096, "llama context size")
	threads := flag.Int("threads", 6, "decode threads")
	flag.Parse()

	if err := llama.Load(*libPath); err != nil {
		fail(err)
	}
	llama.LogSet(llama.LogSilent())
	llama.Init()
	defer llama.Close()

	cases := []evalCase{
		{
			Name:            "warehouse_tool",
			Prompt:          "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?",
			ExpectedToolOut: "1672",
			MaxTokens:       int32(*maxTokens),
			RequireTool:     true,
		},
		{
			Name:            "train_tool",
			Prompt:          "A train leaves City A at 08:00 travelling toward City B at 90 km/h. A second train leaves City B at 09:30 travelling toward City A at 110 km/h. The distance between the cities is 450 km. At what time do the trains meet, and how far from City A is the meeting point?",
			ExpectedToolOut: "11:04:30\n276.75 km from City A",
			MaxTokens:       int32(*maxTokens),
			RequireTool:     true,
		},
		{
			Name:            "pet_logic_tool",
			Prompt:          "Three friends, Emma, Liam, and Priya, each own exactly one pet: a cat, a dog, or a parrot. Emma does not own the cat. Liam does not own the dog. Priya owns a furry pet. The parrot is not owned by Emma. Who owns each pet?",
			ExpectedToolOut: "Emma: dog\nLiam: parrot\nPriya: cat",
			MaxTokens:       int32(*maxTokens),
			RequireTool:     true,
		},
		{
			Name:             "warehouse_final_after_tool",
			Prompt:           "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. In Q2 it restocks 800 units. In Q3 it sells 640 units. How many units remain at the end of Q3?\n\nrun_python result:\n1672\n\nReturn the final answer only.",
			ExpectedFinal:    "1672",
			MaxTokens:        int32(*maxTokens),
			RequireFinalOnly: true,
		},
	}

	for _, spec := range []struct {
		name string
		path string
	}{
		{name: "base", path: *basePath},
		{name: "finetuned", path: *ftPath},
	} {
		if spec.path == "" {
			continue
		}
		m, err := loadModel(spec.name, spec.path, uint32(*nctx), int32(*threads))
		if err != nil {
			fail(err)
		}
		runSuite(m, cases)
		_ = llama.Free(m.ctx)
		_ = llama.ModelFree(m.model)
	}
}

func loadModel(name, path string, nctx uint32, threads int32) (*loadedModel, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	params := llama.ModelDefaultParams()
	params.NGpuLayers = 0
	params.Devices = 0

	model, err := llama.ModelLoadFromFile(abs, params)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = nctx
	ctxParams.NBatch = 1024
	ctxParams.NUbatch = 512
	ctxParams.NThreads = threads
	ctxParams.NThreadsBatch = threads
	ctxParams.Offload_kqv = 0
	ctxParams.OpOffload = 0

	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("init %s: %w", name, err)
	}
	tmpl := llama.ModelChatTemplate(model, "")
	if strings.TrimSpace(tmpl) == "" {
		_ = llama.Free(ctx)
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("%s has no embedded chat template", name)
	}
	return &loadedModel{
		name:         name,
		path:         abs,
		model:        model,
		ctx:          ctx,
		vocab:        llama.ModelGetVocab(model),
		chatTemplate: tmpl,
	}, nil
}

func runSuite(m *loadedModel, cases []evalCase) {
	fmt.Printf("\n== %s ==\n%s\n", m.name, m.path)
	passed := 0
	for _, c := range cases {
		response, err := generate(m, c.Prompt, c.MaxTokens)
		if err != nil {
			fmt.Printf("FAIL %-24s generation error: %v\n", c.Name, err)
			continue
		}
		ok, detail := scoreCase(c, response)
		if ok {
			passed++
			fmt.Printf("PASS %-24s %s\n", c.Name, detail)
		} else {
			fmt.Printf("FAIL %-24s %s\n", c.Name, detail)
			fmt.Printf("     response: %s\n", oneLine(response))
		}
	}
	fmt.Printf("score %d/%d\n", passed, len(cases))
}

func generate(m *loadedModel, userPrompt string, maxTokens int32) (string, error) {
	mem, err := llama.GetMemory(m.ctx)
	if err != nil {
		return "", err
	}
	if err := llama.MemoryClear(mem, true); err != nil {
		return "", err
	}

	prompt, err := template.ApplyWithOptions(
		m.chatTemplate,
		[]message.Message{
			message.Chat{Role: "system", Content: systemPrompt},
			message.Chat{Role: "user", Content: userPrompt},
		},
		true,
		template.Options{EnableThinking: false},
	)
	if err != nil {
		return "", err
	}

	// parseSpecial=true so the chat markup becomes special ids; addSpecial adds bos (the GGUF template omits it).
	tokens := llama.Tokenize(m.vocab, prompt, true, true)
	if len(tokens) == 0 {
		return "", errors.New("empty prompt tokenization")
	}
	batch := llama.BatchGetOne(tokens)
	sampler := llama.SamplerChainInit(llama.SamplerChainDefaultParams())
	defer llama.SamplerFree(sampler)
	llama.SamplerChainAdd(sampler, llama.SamplerInitGreedy())

	var out strings.Builder
	for pos := int32(0); pos < maxTokens; pos += batch.NTokens {
		if code, err := llama.Decode(m.ctx, batch); err != nil || code != 0 {
			if err != nil {
				return out.String(), err
			}
			return out.String(), fmt.Errorf("llama_decode returned %d", code)
		}
		token := llama.SamplerSample(sampler, m.ctx, -1)
		if llama.VocabIsEOG(m.vocab, token) {
			break
		}
		buf := make([]byte, 256)
		n := llama.TokenToPiece(m.vocab, token, buf, 0, true)
		if n > 0 {
			out.Write(buf[:n])
		}
		llama.SamplerAccept(sampler, token)
		batch = llama.BatchGetOne([]llama.Token{token})
	}
	return strings.TrimSpace(out.String()), nil
}

func scoreCase(c evalCase, response string) (bool, string) {
	if c.RequireFinalOnly {
		if normalize(response) == normalize(c.ExpectedFinal) {
			return true, "final answer matched"
		}
		return false, fmt.Sprintf("final mismatch got=%q want=%q", strings.TrimSpace(response), c.ExpectedFinal)
	}

	code := ""
	if m := fenceRe.FindStringSubmatch(response); m != nil {
		code = m[1]
	} else if calls := message.ParseToolCalls(response); len(calls) == 1 && calls[0].Function.Name == "run_python" {
		code = calls[0].Function.Arguments["code"] // legacy JSON contract
	} else {
		return false, "no fenced python block (or legacy tool call) in response"
	}
	if strings.TrimSpace(code) == "" {
		return false, "empty code"
	}
	got, err := runPython(code)
	if err != nil {
		return false, fmt.Sprintf("python failed: %v", err)
	}
	if normalize(got) != normalize(c.ExpectedToolOut) {
		return false, fmt.Sprintf("tool output mismatch got=%q want=%q", got, c.ExpectedToolOut)
	}
	return true, "tool call executed to expected result"
}

func runPython(code string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "python3", "-c", code)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if ctx.Err() == context.DeadlineExceeded {
		return "", ctx.Err()
	}
	if err != nil {
		return strings.TrimSpace(stdout.String()), fmt.Errorf("%w: %s", err, strings.TrimSpace(stderr.String()))
	}
	return strings.TrimSpace(stdout.String()), nil
}

func normalize(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func oneLine(s string) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", `\n`)
	if len(s) > 360 {
		return s[:360] + "..."
	}
	return s
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

var fenceRe = regexp.MustCompile("(?s)```(?:python)?[ \t]*\n(.*?)\n?```")
