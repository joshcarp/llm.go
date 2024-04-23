package main

import (
	"log"
	"net/http"
)

func main() {
	// Define the directory containing static files
	fs := http.FileServer(http.Dir("."))

	// Serve static files
	http.Handle("/", fs)

	// Start the server
	log.Println("Server started on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
