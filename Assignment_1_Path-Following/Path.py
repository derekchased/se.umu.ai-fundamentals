import json

# Load the path from a file and convert it into a list of coordinates
class Path:

    def __init__(self, file_name):
    
        with open(file_name) as path_file:
            data = json.load(path_file)
    
        self.path = data
    
    def getPath(self):
        vecArray = [{'X': p['Pose']['Position']['X'], \
                     'Y': p['Pose']['Position']['Y'], \
                     'Z': p['Pose']['Position']['Z']}\
                     for p in self.path]
        return vecArray
