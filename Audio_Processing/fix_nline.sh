#!/bin/bash

# Usage: ./fix_metadata.sh input.csv > output.csv

awk -F'|' '{
    # 1. Clean up potential Windows carriage returns (\r)
    gsub(/\r/, "", $0)

    # 2. Check if we have fewer than 3 fields
    if (NF < 3) {
        # Print the fragment WITHOUT a newline
        printf "%s", $0
        
        # Grab the next line
        if (getline next_line > 0) {
            # Clean the next line too
            gsub(/\r/, "", next_line)
            # Print the next line with a standard newline
            print next_line
        }
    } else {
        # If the line is already correct, just print it
        print $0
    }
}' metadata_final.csv