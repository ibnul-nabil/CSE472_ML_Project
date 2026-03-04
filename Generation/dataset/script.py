input_path = "metadata.csv"
output_path = "metadata_filtered.csv"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    first_line = infile.readline()
    outfile.write(first_line)
    for line in infile:
        if line.startswith("common_voice_s1"):
            outfile.write(line)