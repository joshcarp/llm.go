package llmgo

import (
	"fmt"
	"io"
	"net/http"
	"os"
)

func downloadFromHF(outputPath, url string) error {
	fmt.Println("Downloading file from Huggingface...")

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to get file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	contentLength := resp.ContentLength
	var totalRead int64 = 0

	out, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", outputPath, err)
	}
	defer out.Close()

	buf := make([]byte, 4096) // Adjust buffer size to your needs
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			totalRead += int64(n)
			percentage := float64(totalRead) / float64(contentLength) * 100
			fmt.Printf("\rDownloading... %.2f%% complete", percentage)

			_, writeErr := out.Write(buf[:n])
			if writeErr != nil {
				return fmt.Errorf("failed to write to file %s: %w", outputPath, writeErr)
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read data: %w", err)
		}
	}
	fmt.Println("\nDownload complete.")
	return nil
}
