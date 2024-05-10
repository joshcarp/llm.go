package llmgo

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
)

// CLI global variables
var (
	cacheDir string
	err      error

	// this is the base folder that consist of basic tokenizer
	// and basic model + its weights
	DEFAULT_BASE_FOLDER_NAME = "base"

	// basic tokenizer and model
	basicTokenizerURL  string = "https://huggingface.co/joshcarp/llm.go/resolve/main/gpt2_tokenizer.bin"
	basicModelURL      string = "https://huggingface.co/joshcarp/llm.go/resolve/main/gpt2_124M.bin"
	basicModelDebugURL string = "https://huggingface.co/joshcarp/llm.go/resolve/main/gpt2_124M_debug_state.bin"
)

// initializeCacheDir initializes the cache directory
func initializeCacheDir() {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	cacheDir = homeDir + "/.cache/llmgo"
	os.MkdirAll(cacheDir, os.ModePerm) // Ensure the directory exists
}

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "llmgo",
	Short: "CLI tool to interface with GPT-2 model",
	Long: `
		This CLI tool provides a direct interface to the GPT-2 model, allowing users to generate text, train new models, and fine-tune existing models directly from the command line. It is designed for both developers and researchers interested in exploring the capabilities of GPT-2 without the need for complex programming environments.
	`,
}

var initCmd = &cobra.Command{
	Use:   "init",
	Short: "Initialize the GPT-2",
	Long:  `This command loads tokenizer and base model from Huggingface, intializes internal variables for use on inference/training`,
	Run: func(cmd *cobra.Command, args []string) {
		// check whether there's base configuration directory.
		// if not, create it
		baseConfigPath := filepath.Join(cacheDir, DEFAULT_BASE_FOLDER_NAME)
		files, err := os.ReadDir(baseConfigPath)
		if err != nil {
			// create the base config file
			if err := os.MkdirAll(baseConfigPath, os.ModePerm); err != nil {
				fmt.Println("failed to create base configuration directory: %w", err)
				return
			}
		}

		// If somehow the base configuration directory is created,
		// but if any of the base files is not found, download them
		tokenizerExistAndValid := false
		modelExistAndValid := false
		modelDebugStateExistAndValid := false

		// list files in the cache directory
		// if the tokenizer/model is not in the cache
		// download it from Huggingface
		for _, file := range files {
			if file.Name() == filepath.Base(basicTokenizerURL) {
				_, err = NewTokenizer(filepath.Join(cacheDir, DEFAULT_BASE_FOLDER_NAME, file.Name()))
				if err != nil {
					continue
				}
				tokenizerExistAndValid = true
			}

			if file.Name() == filepath.Base(basicModelURL) {
				// TODO: validate the model
				modelExistAndValid = true
			}

			if file.Name() == filepath.Base(basicModelDebugURL) {
				// TODO: validate the model debug state file
				modelDebugStateExistAndValid = true
			}
		}

		if !tokenizerExistAndValid {
			fmt.Println("Tokenizer not found, downloading...")
			if err := downloadFromHF(filepath.Join(baseConfigPath, filepath.Base(basicTokenizerURL)), basicTokenizerURL); err != nil {
				fmt.Println("failed to download tokenizer: %w", err)
				return
			}
		}

		if !modelExistAndValid {
			fmt.Println("Model not found, downloading...")
			if err := downloadFromHF(filepath.Join(baseConfigPath, filepath.Base(basicModelURL)), basicModelURL); err != nil {
				fmt.Println("failed to download model: %w", err)
				return
			}
		}

		if !modelDebugStateExistAndValid {
			fmt.Println("Model debug state not found, downloading...")
			if err := downloadFromHF(filepath.Join(baseConfigPath, filepath.Base(basicModelDebugURL)), basicModelDebugURL); err != nil {
				fmt.Println("failed to download model debug state: %w", err)
				return
			}
		}
	},
}

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run GPT inference",
	Long:  `This command initiates the GPT inference process. It allows users to input text and receive generated continuations based on the selected model. This feature is particularly useful for generating text, experimenting with AI-driven content creation, and more.`,
	// Run: func(cmd *cobra.Command, args []string) {
	// 	fmt.Println("run called")
	// },
}

// gpt2Cmd represents the gpt2 command
var gpt2Cmd = &cobra.Command{
	Use:   "gpt2",
	Short: "Run GPT-2 inference",
	Long:  `This command specifically initiates the GPT-2 inference process. It allows users to input text and receive AI-generated text continuations based on the GPT-2 model.`,
	Run: func(cmd *cobra.Command, args []string) {
		// load tokenizer
		// TODO: custom configuration
		tok, err := NewTokenizer(filepath.Join(cacheDir, DEFAULT_BASE_FOLDER_NAME, filepath.Base(basicTokenizerURL)))
		if err != nil {
			fmt.Println("failed to load tokenizer: %w", err)
			fmt.Println("did you forget to run `llmgo init`?")
			return
		}

		for {
			fmt.Printf(">>> ")
			inputReader := bufio.NewReader(os.Stdin)

			input, err := inputReader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					fmt.Println("\nExiting command. Thanks for using llmgo!")
					break
				}
				fmt.Println("Failed to read input:", err)
				continue
			}

			// tokenize the input
			tokens, err := tok.Encode(input)
			if err != nil {
				fmt.Println("Failed to tokenize input:", err)
				continue
			}

			// slowly print the input, simulating the typing effect
			for _, t := range tokens {
				fmt.Print(tok.tokenTable[t])
				time.Sleep(100 * time.Millisecond)
			}
			fmt.Println()
		}
	},
}

func InitializeCommand() {
	initializeCacheDir()

	rootCmd.AddCommand(runCmd)
	rootCmd.AddCommand(initCmd)
	runCmd.AddCommand(gpt2Cmd)

	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}

	// Here you will define your flags and configuration settings.

	// Cobra supports Persistent Flags which will work for this command
	// and all subcommands, e.g.:
	// gpt2Cmd.PersistentFlags().String("foo", "", "A help for foo")

	// Cobra supports local flags which will only run when this command
	// is called directly, e.g.:
	// gpt2Cmd.Flags().BoolP("toggle", "t", false, "Help message for toggle")
}
