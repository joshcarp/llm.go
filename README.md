<h1 >llm.go</h1>

<div >

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/joshcarp/llm.go)](https://github.com/joshcarp/grpctl/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/joshcarp/llm.go)](https://github.com/joshcarp/grpctl/pulls)
[![License](https://img.shields.io/badge/license-apache2-blue.svg)](/LICENSE)

</div>

GPT-2 implementation written in go only using the standard library.

## 🪞 Quick start

Install python dependencies, output tokenized dataset

```bash
make setup
```

Run the training script:
```bash
make train
```

This will run `go run ./cmd/traingpt2/main.go`

Run the testing script:
```bash
make test
```

This will run `go run ./cmd/testgpt2/main.go`

# TODO
- [x] Tokenize input text (Needed for WASM)
- [ ] Very slow, need to improve performance.
- [ ] It runs in WASM but using WebGPU bindings might be fun.
- [ ] More refactoring.
- [ ] Running as CLI.

# 🖋️ License <a name = "license"></a>

See [LICENSE](LICENSE) for more details.

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- 
- This is a fork of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) written in pure go.
 