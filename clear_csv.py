import string
import argparse


def main():
    parser = argparse.ArgumentParser(description='Removes weird characters from input file.')
    parser.add_argument('-f', '--file', required=True, dest='file_name', help='Path to the input file')

    args = parser.parse_args()
    infile_name = args.file_name
    output_file_name = infile_name.replace('.', '_clean.')
    try:
        with open(infile_name, 'r', encoding="utf8") as infile, open(output_file_name, 'w') as outfile:
            infile_content = infile.read()
            cleared_content = ''.join(filter(lambda x: x in string.printable, infile_content))
            outfile.write(cleared_content)

    except PermissionError as pe:
        print('Operation Failed!\n{}, make sure the file is not being used.'.format(str(pe).replace('\\\\', '\\')))
    except FileNotFoundError as fnfe:
        print('Operation Failed!\n{}'.format(str(fnfe).replace('\\\\', '\\')))


if __name__ == "__main__":
    main()
