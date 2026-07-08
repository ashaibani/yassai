package main

import _ "embed"

//go:embed index.html
var indexHTML string

//go:embed app.js
var appJS string
