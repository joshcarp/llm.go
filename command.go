package llmgo

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"
)

var tok Tokenizer
var err error

// rootCmd represents the base command when called without any subcommands
var rootCmd = &cobra.Command{
	Use:   "llmgo",
	Short: "CLI tool to interface with GPT-2 model",
	Long: `
		This CLI tool provides a direct interface to the GPT-2 model, allowing users to generate text, train new models, and fine-tune existing models directly from the command line. It is designed for both developers and researchers interested in exploring the capabilities of GPT-2 without the need for complex programming environments.
	`,
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
	tok, err = NewTokenizer("./gpt2_tokenizer.bin")
	if err != nil {
		panic(err)
	}

	rootCmd.AddCommand(runCmd)
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
