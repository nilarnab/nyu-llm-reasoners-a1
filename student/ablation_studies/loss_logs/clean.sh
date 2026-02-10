#!/bin/bash

# Usage: ./filter_csv.sh input.csv > output.csv

awk -F',' '$3 != "_"' "$1"