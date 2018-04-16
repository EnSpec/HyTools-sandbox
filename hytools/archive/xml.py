import xml.etree.cElementTree as ET


basename = "TEST"

def h5_structure(basename):


root = ET.Element(basename)

reflectance = ET.SubElement(root, "Reflectance")
metadata = ET.SubElement(reflectance, "Metadata")

anc_imagery = ET.SubElement(metadata, "Ancillary_Imagery")
coord_sys = ET.SubElement(metadata, "Coordinate_System")
logs = ET.SubElement(metadata, "Logs")
spectral_data = ET.SubElement(metadata, "Spectral_Data")





test_dir = '%s/Dropbox/projects/hyTools/HyTools-sandbox/testing' % home

tree.write("%s/filename.xml" % test_dir)