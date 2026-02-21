import xml.etree.ElementTree as ET

def indent(elem, level=0):
    """Add whitespace for pretty‐printing XML."""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level+1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

# 1) Build the <additional> root
additional = ET.Element("additional")

# 2) Define each vType entry
vtypes = [
    ("Car",           {"vClass":"passenger",  "color":".8,.2,.2",   "carFollowModel":"Krauss"}),
    ("Bus",           {"vClass":"bus",        "carFollowModel":"Krauss", "speedFactor":"1.4"}),
    ("Medium", {"vClass":"passenger",  "carFollowModel":"Krauss"}),
    ("Heavy",  {"vClass":"truck",      "carFollowModel":"Krauss"}),
    ("Motorcycle",    {"vClass":"motorcycle", "carFollowModel":"Krauss"}),
    ("Taxi",          {"vClass":"taxi",       "carFollowModel":"Krauss"}),
]

# 3) Append each <vType> element
for vid, attrs in vtypes:
    ET.SubElement(additional, "vType", id=vid, **attrs)

# 4) Pretty‐print indentation
indent(additional)

# 5) Write to osm.trip.xml
tree = ET.ElementTree(additional)
tree.write("osm.type.xml", encoding="utf-8", xml_declaration=True)
print("osm.type.xml populated with <additional> / <vType> definitions.")
