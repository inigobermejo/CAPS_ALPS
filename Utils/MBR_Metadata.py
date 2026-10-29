from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

mrb_dir = Path("C:/Users/lucp13173/Documents/Data/CAPS/Jessa/Segmented")

for mrb_file in sorted(mrb_dir.glob("*.mrb")):

    print("\n" + "=" * 100)
    print(f"FILE: {mrb_file.name}")
    print("=" * 100)

    with zipfile.ZipFile(mrb_file, "r") as z:

        mrml_files = [
            f for f in z.namelist()
            if f.lower().endswith(".mrml")
        ]

        if not mrml_files:
            print("No MRML scene found")
            continue

        root = ET.fromstring(z.read(mrml_files[0]))

        # Print segmentation nodes
        print("\nSEGMENTATION NODES")
        print("-" * 80)

        for node in root.iter():
            if "Segmentation" in node.tag:
                print("\nTAG:", node.tag)
                for key, value in node.attrib.items():
                    print(f"  {key}: {value}")

        # Print label-map / volume nodes
        print("\nVOLUME / LABELMAP NODES")
        print("-" * 80)

        for node in root.iter():
            if "Volume" in node.tag or "LabelMap" in node.tag:
                print("\nTAG:", node.tag)
                for key, value in node.attrib.items():
                    print(f"  {key}: {value}")