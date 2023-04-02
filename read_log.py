'''
Example of reading text data - in this case from a log file meanfit_03_06.log.
'''

import numpy as np
import argparse
import IPython

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="Log filename to load.")

def main(filename):
    # The "with" block automatically handles closing of a file stream object
    # Files can be opened for read-only (default), writing (which deletes any 
    #   existing contents first or creates a new file), and appending.
    with open(filename, "r") as f:
        params = []
        SE_list = []
        SE_total = []
        linecont = None
        # you can loop through lines in a text file by looping over the file obj
        for line in f:
            line = line.rstrip() # remove trailing whitespace
            # Note: There is a lot of string manipulation here, and there is a 
            #   LOT to learn in string manipulation. The main thing to know is
            #   that there are a lot of operations that can be done on strings,
            #   and you can read about them in the Python docs under Built-in 
            #   Types: str. See https://docs.python.org/3/library/stdtypes.html

            # deal with wrapped lines
            if linecont is not None:
                line = linecont+line
                linecont = None
            # grab parameter values. Sometimes, the line wrapped. This can be 
            #   checked by seeing if the brackets were closed at the end.
            if line[0] == '[':
                if line[-1] != ']':
                    linecont = line
                    continue # this goes directly to the next iteration of the loop.
                params.append(np.array(line.strip('][').split(), dtype=float))
            # grab error
            elif line[:21] == 'SE, per SPAHR class: ':
                if line[-1] != ']':
                    linecont = line
                    continue
                SE_list.append(np.array(line[21:].strip('][').split(','), dtype=float))
            elif line[:11] == 'SE, total: ':
                SE_total.append(float(line[11:]))
    
    # The file object is now closed because the with block is done.

    params = np.array(params)
    SE_list = np.array(SE_list)
    SE = np.array(SE_total)
    # Now that we have everything we need, start up IPython for further analysis
    IPython.embed()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.filename)