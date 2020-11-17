#Christopher Yun
#@christopheryun72@berkeley.edu

import os
import selenium
from selenium import webdriver
import time
from PIL import Image
import io
import requests
from selenium.common.exceptions import ElementClickInterceptedException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.headless = True
options.add_argument("--window-size=1920,1200")

DRIVER_PATH = '/Users/christopheryun/PCS/chromedriver'
driver = webdriver.Chrome(options=options, executable_path=DRIVER_PATH)

search_url = 'https://www.google.com/search?q=clarence+thomas+portrait&tbm=isch&ved=2ahUKEwihkeaIl4btAhWwKzQIHSKCC1EQ2-cCegQIABAA&oq=clarence+thomas+portrait&gs_lcp=CgNpbWcQAzICCAA6BQgAELEDOgQIABBDOgQIABAYULwmWJ83YI84aABwAHgAgAFsiAHUBZIBAzguMZgBAKABAaoBC2d3cy13aXotaW1nwAEB&sclient=img&ei=v_mxX-G1G7DX0PEPooSuiAU&bih=1431&biw=973&rlz=1C5CHFA_enUS852US852'
driver.get(search_url.format(q='Clarence Thomas Face'))
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(5)#sleep_between_interactions
imgResults = driver.find_elements_by_xpath("//img[contains(@class,'Q4LuWd')]")
totalResults=len(imgResults)
print(imgResults)
img_urls = set()
for i in range(0,totalResults):
	print(i)
	img=imgResults[i]
	try:
		img.click()
		time.sleep(2)
		actual_images = driver.find_elements_by_css_selector('img.n3VNCb')
		for actual_image in actual_images:
			if actual_image.get_attribute('src') and 'https' in actual_image.get_attribute('src'):
				img_urls.add(actual_image.get_attribute('src'))
	except:
		break
print(img_urls)
baseDir = os.getcwd()
for i, url in enumerate(img_urls):
	file_name = f"{i:150}pt2.jpg"
	try: 
		image_content = requests.get(url).content
	except Exception as exception:
		print(f"Error, Could not download {url} - {exception}")
	try:
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file).convert('RGB')
		file_path = os.path.join(baseDir + '/ClarenceThomas/', file_name.strip())
		with open(file_path, 'wb') as f:
			image.save(f, "JPEG", quality=85)
		print(f"SAVED - {url} - AT: {file_path}")
	except Exception as exception:
		print(f"Error2, Could not download {url} - {exception}")