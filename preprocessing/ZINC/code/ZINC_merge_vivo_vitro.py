file1 = "ZINC\\new2\\without_common_in_vitro_extracted_zinc_ids_new.csv"
file2 = "ZINC\\new2\\in_vivo_extracted_zinc_ids_new.csv"
output_file = "ZINC\\new2\\new_ZINC_ids_labelled.csv"

header = "ZINC,label\n"

with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file, 'w') as output:
    output.write(header)
    
    # Scrivi gli elementi del primo file con etichetta 0
    for line in f1:
        output.write(f"{line.strip()},0\n")
    
    # Scrivi gli elementi del secondo file con etichetta 1
    for line in f2:
        output.write(f"{line.strip()},1\n")