#!/usr/bin/env bash

if (which inkscape > /dev/null); then
  for filename in assets/gen/*.svg; do
    out_path="${filename%.svg}.png"
    inkscape "${filename}" -o "${out_path}" -w 4096
    abs_path=$(realpath "${out_path}")
    echo Converted "file://${abs_path}"
  done
else
  echo "Warning: 'inkscape' not in PATH. Skipping conversion of SVG images to PNG images."
fi
