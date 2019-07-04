package util

import (
	"os"
)

func WriteDotToFile(dot string) error {
	file, err := os.Create("graph.dot")
	defer file.Close()
	if err != nil {
		return err
	}
	_, err = file.WriteString(dot)
	return err
}
