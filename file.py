import czifile
from xml.etree import ElementTree as ET

czi_path = r"C:\Users\zindi\PycharmProjects\P2\unpacked images\Raw\WT_NADA_RADA_HADA_NHS_40min_ROI1_SIM.czi"

with czifile.CziFile(czi_path) as czi:
    # Get and parse the XML metadata
    metadata_xml = czi.metadata()
    root = ET.fromstring(metadata_xml)

    # Extract pixel sizes
    scaling = root.find(".//Scaling")
    pixel_sizes = {}
    for distance in scaling.findall(".//Distance"):
        dim = distance.get('Id')
        value = distance.find('Value').text
        pixel_sizes[dim] = f"{float(value) * 1e6:.4f} µm"  # Convert to µm

    # Extract microscope information
    microscope_info = {
        'Model': root.findtext(".//Microscope/System", default="N/A"),
        'Objective': root.findtext(".//Objective/Manufacturer/Model", default="N/A"),
        'NA': root.findtext(".//Objective/LensNA", default="N/A"),
        'Magnification': root.findtext(".//Objective/NominalMagnification", default="N/A"),
        'Immersion': root.findtext(".//Objective/Immersion", default="N/A")
    }

# Print results
print("=== Pixel Size ===")
for dim, size in pixel_sizes.items():
    print(f"{dim}: {size}")

print("\n=== Microscope Information ===")
for key, value in microscope_info.items():
    print(f"{key}: {value}")