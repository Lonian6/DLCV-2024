from plyfile import PlyData
import sys


ply_path = sys.argv[1]
ply_struct = PlyData.read(ply_path)

for element in ply_struct.elements:
    print("Element name:", element.name)
    print("Number of elements:", len(element.data))
    print(len(element.data[0]))
    print(element.data[0])