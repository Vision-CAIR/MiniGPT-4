import ijson

# specify input and output file paths
input_file = 'ccs_synthetic_filtered_large.json'
output_file = 'ccs_synthetic_filtered_large.tsv'

# set header to None
headers = None

# load JSON data from input file and open the output file at same time
with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
    objects = ijson.items(in_file, 'item')
    
    for obj in objects:
        # extract header and data from JSON
        if headers is None:
            headers = list(obj.keys())
            out_file.write('\t'.join(headers) + '\n')
        
        # write data to TSV file line by line
        row = '\t'.join(str(obj[key]) for key in headers)
        out_file.write(row + '\n')
