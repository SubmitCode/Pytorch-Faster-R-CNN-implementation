import xml.etree.ElementTree as et

path = 'VOCdevkit/VOC2007/Annotations/000005.xml'

root = et.parse(path).getroot()
for obj in root.findall('object'):
    for item in obj.getchildren():
        print(f"tag={item.tag} text={item.text}")
    print()
    print()