import pandas as pd
import xml.etree.ElementTree as ET


def apply_best_to_xml(history_csv='optimization_history_nevergrad_40_40_10_10.csv', xml_file='osm.type.xml'):
    """
    Reads optimization_history.csv, finds the row with the minimum objective,
    and writes its parameters into osm.type.xml.
    """
    # Load history
    df = pd.read_csv(history_csv)
    if 'objective' not in df.columns:
        raise ValueError("CSV must contain an 'objective' column")
    # Find best row
    best_row = df.loc[df['objective'].idxmin()]
    # Extract parameter columns (all except 'objective')
    params = best_row.drop('objective').to_dict()

    # Parse XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Update attributes
    for v in root.findall(".//vType"):
        for key, val in params.items():
            val_str = str(val)
            if "_" in key:
                # composite key: "<VehicleType>_<param>"
                vt, param = key.split("_", 1)
                if v.get("id") == vt:
                    v.set(param, val_str)
            else:
                # global parameter: set on every vType
                v.set(key, val_str)

    # Write back
    tree.write(xml_file)
    print(f"Updated {xml_file} with best parameters from {history_csv}")

if __name__ == "__main__":
    apply_best_to_xml()
