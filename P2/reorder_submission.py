import csv
import sys

if len(sys.argv) != 3:
    print("USAGE: python reorder_submission.py FILE_TO_REORDER.csv NEW_FILE_NAME.csv")
    sys.exit()

file_to_reorder = sys.argv[1]
newfile_name = sys.argv[2]

# READ IN KEYS IN CORRECT ORDER AS LIST
with open('keys.csv','r') as f:
    keyreader = csv.reader(f)
    keys = [key[0] for key in keyreader]

# READ IN ALL PREDICTIONS, REGARDLESS OF ORDER
with open(file_to_reorder) as f:
    oldfile_reader = csv.reader(f)
    D = {}
    for i,row in enumerate(oldfile_reader):
        if i == 0:
            continue
        _id, pred = row
        D[_id] = pred

# WRITE PREDICTIONS IN NEW ORDER
with open(newfile_name,'w') as f:
    writer = csv.writer(f)
    writer.writerow(('Id','Prediction'))
    for key in keys:
        writer.writerow((key,D[key]))

print("".join(["Reordered ", file_to_reorder," and wrote to ", newfile_name]))
