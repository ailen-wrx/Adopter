#!/bin/bash

# Check if a filename is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

# Loop through each line in the file
while IFS= read -r line; do
  sum=0
  # Split the line into fields using space or tab as the delimiter
  fields=($line)
  # Iterate through fields starting from the second one
  for ((i = 1; i < ${#fields[@]}; i++)); do
    field="${fields[i]}"
    # Check if the field is a number (handles positive and negative numbers)
    if [[ $field =~ ^-?[0-9]+(\.[0-9]+)?$ ]]; then
      sum=$(awk "BEGIN {print $sum + $field}")
    fi
  done
  # print the first column of this line
  echo -n "${fields[0]}: "
  echo "$sum"
done < "$1"