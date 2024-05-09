/*
Copyright © 2024 NAME HERE <EMAIL ADDRESS>
*/
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
	Short: "A brief description of your application",
	Long: `A longer description that spans multiple lines and likely contains
examples and usage of using your application. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	// Uncomment the following line if your bare application
	// has an action associated with it:
	// Run: func(cmd *cobra.Command, args []string) { },
}

// runCmd represents the run command
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	// Run: func(cmd *cobra.Command, args []string) {
	// 	fmt.Println("run called")
	// },
}

// gpt2Cmd represents the gpt2 command
var gpt2Cmd = &cobra.Command{
	Use:   "gpt2",
	Short: "A brief description of your command",
	Long: `A longer description that spans multiple lines and likely contains examples
and usage of using your command. For example:

Cobra is a CLI library for Go that empowers applications.
This application is a tool to generate the needed files
to quickly create a Cobra application.`,
	Run: func(cmd *cobra.Command, args []string) {
		for {
			fmt.Printf(">>> ")
			inputReader := bufio.NewReader(os.Stdin)

			input, err := inputReader.ReadString('\n')
			if err != nil {
				if err == io.EOF {
					fmt.Println("\nExiting command. Thanks for using llmgo!")
					os.Exit(0)
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
