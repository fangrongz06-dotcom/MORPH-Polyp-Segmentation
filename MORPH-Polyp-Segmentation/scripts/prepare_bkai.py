"""Index the labeled BKAI external evaluation pool with prepared binary GT."""
import argparse
import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.dataset import read_binary_mask
from scripts.prepare_data import EXTENSIONS, write_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=ROOT/"splits/local/bkai.csv")
    args = parser.parse_args()
    directory = args.data_root/"TestDataset/BKAI"
    def index(folder):
        files = [p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS]
        mapping = {p.stem:p for p in files}
        if len(mapping)!=len(files): raise ValueError("Ambiguous filename stems")
        return mapping
    images,masks = index(directory/"images"),index(directory/"masks")
    if len(images)!=1000 or set(images)!=set(masks):
        raise ValueError("Expected 1000 labeled image/GT pairs from the official training partition")
    rows = []
    for image_id,image_path in sorted(images.items()):
        with Image.open(image_path) as image, Image.open(masks[image_id]) as mask:
            if image.size!=mask.size: raise ValueError("Image/GT geometry differs")
            read_binary_mask(mask,"bkai")
        rows.append({"dataset":"bkai","filename":image_path.name,"mask_filename":masks[image_id].name})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_split(args.output,rows)
    print("Indexed 1000 original BKAI records; audit content groups before statistical inference")


if __name__=="__main__":
    main()
