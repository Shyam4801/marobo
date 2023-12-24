import csv, pickle

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
