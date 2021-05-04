import util

if __name__ == "__main__":
    dataset_dir = '../imagenet/val'
    out_file = '../masks/eccentricity_val.txt'
    other_mask = '../masks/val.txt'

    valid, invalid = util.filter_dataset(dataset_dir, util.is_monochrome, invert=True, log_period=25)

    with open(out_file, 'w') as f:
        for i in valid:
            f.write(str(i))

    grayscale = []

    with open(other_mask, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            grayscale.append(int(line))

    grayscale_s = set(grayscale)
    monochrome_s = set(valid)
    diff = grayscale_s - monochrome_s
    print(len(diff))
    diff = monochrome_s - grayscale_s
    print(len(diff))




