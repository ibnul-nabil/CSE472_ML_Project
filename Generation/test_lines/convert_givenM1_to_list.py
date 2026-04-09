# Script to convert givenM1.txt to a Python list of strings format like cgpt100.txt

def convert_to_list_format(input_path, output_path, var_name="givenM1_texts"):
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    # Extract the text after the first '|' in each line, strip whitespace
    texts = [line.split('|', 1)[1].strip() for line in lines if '|' in line]
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"{var_name} = [\n")
        for text in texts:
            outfile.write(f'    "{text}",\n')
        outfile.write("]\n")

if __name__ == "__main__":
    convert_to_list_format(
        "Generation/test_lines/givenM1.txt",
        "Generation/test_lines/givenM1_list.py"
    )
