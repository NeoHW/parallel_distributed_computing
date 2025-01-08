#!/bin/bash

# Downloads the .fasta file from ncbi using the accession number
# Usage: ./download_fasta.sh <accession_number>

# Check if the user has provided the accession number
if [ $# -eq 0 ]; then
    echo "Please provide the accession number (e.g., https://www.ncbi.nlm.nih.gov/nuccore/<get accession number from here>)"
    exit 1
fi

echo "Downloading the .fasta file from NCBI for accession number $1" >&2

if [ -t 1 ]; then
    echo "Saving to file $1.fasta" >&2
    curl -s "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id=$1&db=nuccore&report=fasta" | tee $1.fasta
else
    curl -s "https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id=$1&db=nuccore&report=fasta"
fi

echo "Download complete!" >&2