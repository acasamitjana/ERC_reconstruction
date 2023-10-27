import json
import pdb

LUT = '/home/acasamitjana/Data/BT/AllenAtlasLUT_xtra.txt'
output_file = '/home/acasamitjana/Data/BT/AllenAtlasLUT_xtra.json'
to_write = {}
with open(LUT, 'r') as txtfile:
    rows = txtfile.readlines()
    for r in rows:
        r = r.replace('    ', ' ')
        r = r.replace('   ', ' ')
        r = r.replace('  ', ' ')
        info = r.split(' ')

        print(info)
        colors = [info[2], info[3], info[4], info[5]]

        l_dict = {
            'labelName': info[1].replace(' ', ''),
            'r': int(colors[0]),
            'g': int(colors[1]),
            'b': int(colors[2]),
            'a': int(colors[3].split('\n')[0])
        }
        to_write[info[0]] = l_dict

json_object = json.dumps(to_write, indent=4)
with open(output_file, "w") as outfile:
    outfile.write(json_object)
#
LUT = '/home/acasamitjana/Data/BT/FreeSurferColorLUT.txt'
output_file = '/home/acasamitjana/Data/BT/FreeSurferColorLUT.json'
to_write = {}
with open(LUT, 'r') as txtfile:
    rows = txtfile.readlines()
    for r in rows:
        r = r.replace('     ', ' ')
        r = r.replace('    ', ' ')
        r = r.replace('   ', ' ')
        r = r.replace('  ', ' ')
        info = r.split(' ')

        if len(info) < 4: continue
        try:
            info[0] = int(info[0])
            colors = [info[2], info[3], info[4], info[5].split('\n')[0]]

            l_dict = {
                'labelName': info[1],
                'r': int(colors[0]),
                'g': int(colors[1]),
                'b': int(colors[2]),
                'a': int(colors[3].split('\n')[0])
            }
            to_write[info[0]] = l_dict
            print(info)

        except:
            continue

json_object = json.dumps(to_write, indent=4)
with open(output_file, "w") as outfile:
    outfile.write(json_object)