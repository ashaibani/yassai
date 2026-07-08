package main

import (
	"github.com/ashaibani/yassai/internal/taskclf"
)

func newClassifier(dir, lib string) (*taskclf.Classifier, error) {
	return taskclf.New(dir, "", lib)
}
