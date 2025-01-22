import argparse
from processor_pdf import merge_pdfs_text
from logger import logger

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", help="Pdfs as dataset",required=True)
    return parser.parse_args()


def main():
    args = get_args()
    logger.log(f"Merging pdfs: {args.dataset}")
    text = merge_pdfs_text(pdfs_paths=args.dataset)
    logger.log(f"Total characters: {len(text)}")

    



    

if __name__ == "__main__":
    main()

