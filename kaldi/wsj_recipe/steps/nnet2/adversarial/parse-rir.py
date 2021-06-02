import sys

def main():

    if len(sys.argv) < 5:
        print("Wrong or not enough arguments")
        sys.exit()

    rir_list = sys.argv[1]
    rir_file = sys.argv[2]
    iter = int(sys.argv[3]) - 1
    max_itr = int(sys.argv[4])

    # read in rir with mod to avoid overflows
    with open(rir_list, 'r') as f:
        lines = f.readlines() 
        print('Define RIR layer {}'.format(iter % len(lines)))
        rir = lines[iter % len(lines)]

    # write to rir layer description file
    with open(rir_file, 'w') as f:
        f.write('<ConvRIRComponent> <InputDim> 256 <RIRcoefficients> [ {} ]</ConvRIRComponent>\n'.format(rir))

if __name__ == "__main__":
    main()