import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageColor

#hsl(hue, saturation%, lightness%)

out = Image.new("RGB", (1000, 500), (255, 255, 255))

# get a font
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
# get a drawing context
d = ImageDraw.Draw(out)

#hue = np.linspace(0, 360, 15)
hue = [0, 25, 51, 70, 95, 140, 170, 190, 210, 230, 260, 282, 308, 334, 360]
sat = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 50]
H = []
for i, x in enumerate(np.linspace(10, 900, 15)):
	color = f"hsl({hue[i]}, {sat[i]}%, 50%)"
	d.rectangle([(x, 10),(x+50, 50)], outline=color, fill=color, width=4)
	# draw multiline text
	d.text((x+10, 55), f"{int(hue[i])}", font=fnt, fill=(0, 0, 0))
	H.append(int(hue[i]))

out.show()
print(H)
