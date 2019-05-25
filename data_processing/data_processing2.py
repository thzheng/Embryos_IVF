from PIL import Image
import pytesseract
import csv

img = Image.open(
	"../../EmbryoScopeAnnotatedData/Folder 1/D2019.03.12_S00011_I3205_P_WELL01/D2019.03.12_S00011_I3205_P_WELL01_RUN892.JPG"
	).crop((726,770, 790, 790))
img.save("../../EmbryoScopeAnnotatedData/Folder 1/try.jpg")
text = pytesseract.image_to_string(img ,config ='--psm 6')
# float(text)
# print(float(text))
print(text)

def GetTimeFromImg():
	pass

"""
yield all images with the original H,W and always 3 channel, and the label, folder and well
(the 4th channel is always 255 for a fully saturated png, so we don't need it) 
"""
def yield_image(path, debug=0):
	pass


yield_image("../../EmbryoScopeAnnotatedData")