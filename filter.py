import sys
from util import filter

data_directory = sys.argv[1]
out_file = sys.argv[2]

valid, invalid = filter.filter_dataset(data_directory, filter.is_grayscale, invert=True, log_period=50)

with open(out_file, 'w') as file:
    for i in valid:
        file.write(f'{i}\n')

