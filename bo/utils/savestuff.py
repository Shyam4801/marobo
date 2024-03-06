import csv, pickle, json

def writetocsv(name, rows):
    with open(name, 'a') as f:
        
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(rows)

def save_node(node, filename):
    with open(filename, 'wb') as file:
        try:
            pickle.dump(node, file)
        except AttributeError:
            print('exception ?')
            pass

import sys

def savetotxt(name, stmt):
    # Open a file in write mode
    with open(name, 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f
        
        # Your print statements here
        print(stmt)
        
        # Restore stdout to its original value
        sys.stdout = sys.__stdout__


def savetojson(file_path, data):
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)