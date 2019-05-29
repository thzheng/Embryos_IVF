from PIL import Image
import pytesseract
import csv
import re
import os
import pandas as pd


time_name = ["tPB2","tPNa","tPNf","t2","t3","t4","t5","t6","t7","t8","t9","tSC","tM","tSB","tB","tEB","tHB","tDead"]

def GetTimeFromImg(path):
	img = Image.open(path).crop((727,770, 790, 790))
	# img.save("../../EmbryoScopeAnnotatedData/Folder 1/try.jpg")
	text = pytesseract.image_to_string(img ,config ='--psm 6')
	start_ind = 0
	for i,ch in enumerate(text):
		if ch.isdigit(): 
			start_ind = i
			break
	new_text = text[i:]
	return GetTimeFromText(new_text)


def GetTimeFromText(text = "", debug=0):
	if text is "": return None
	m = re.search(r'.*(?=h)', file_path)
	if m:
		label = m.group(0).strip()
		try:
			res = float(label)
			return res
		except:
			if debug>0: print ("Error finding the label")
			return None
	else: 
		return None

# dictionary with (folder number, well number), and the value as list of (start time, ending time, label)
def GetLabelFromTime(time, file_path, dct = {}):
	m = re.search(r'(?<=/Folder).*(?=/)', file_path)
	n = re.search(r'(?<=WELL).*(?=/)', file_path)
	if m and n:
		folder, well = m.group(0).strip(), n.group(0).strip()
		try:
			if (nt(folder),int(well)) in dct:
				times = dct[int(folder),int(well)]
				for st,et,label in times.items():
					if time>=st and time<et: 
					# if less than the ending time return label.
						return label
		except:
			if debug>0: print ("Error finding the label")


# res is a dictionary storing the mapping from folder_num, well_num to a s list of (start time, ending time, label)
def ProcessAllCSVs(folder_num, path,res):
	annots = pd.read_csv('data.csv')
	for row in annots.rows:
		if row["well"]:
			well_num = int(row["well"])
		for label_header,i in enmerate(time_name):
			# find the ending time
			if not row[lebel_header]: continue
			start_time = float(row[label_header])
			next_time = float("inf")
			for next_time in time_name[i:]:
				if row[next_time]:
					next_time = float(row[next_time])
					break
			if start_time and next_time:
				res[folder_num, well_num].append(label_header,start_time,end_time)










# def yield_image(path, debug=0):
# 	pass
# yield_image("../../EmbryoScopeAnnotatedData")

print(GetTimeFromImg('../../test_EmbryoScopeAnnotatedData/Folder 11/D2019.03.16_S00014_I3205_P_WELL02/D2019.03.16_S00014_I3205_P_WELL02_RUN016.JPG'))

path = "../../test_EmbryoScopeAnnotatedData/"
counter = 0
for directory in os.listdir(path):
	if directory[0] is ".": continue
	file_path = os.path.join(path, directory)
	for well in os.listdir(file_path):
		if well[0] is ".": continue
		full_path_to_well = os.path.join(file_path, well)
		for img in os.listdir(full_path_to_well):
			if img[0] is ".": continue
			print(counter)
			full_path_to_img = os.path.join(full_path_to_well, img)
			print("current_image: " + full_path_to_img)
			if GetTimeFromImg(full_path_to_img):
				print(GetTimeFromImg(full_path_to_img))
			else:
				print("error")