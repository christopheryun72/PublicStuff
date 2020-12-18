#Christopher Yun
#christopheryun72@berkeley.edu
"""
https://www.dickssportinggoods.com/f/total-footwear-markdowns?filterFacets=facetStore%3AISA%2CSHIP

Reqs: Collect product !name, !url, !price, stock (couldn't find, so doing discount instead) and save to database; save a set of time series: save the data every 5 mins
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

site = 'https://www.dickssportinggoods.com/f/total-footwear-markdowns?filterFacets=facetStore%3AISA%2CSHIP'
homepage = 'https://www.dickssportinggoods.com'
header = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Upgrade-Insecure-Requests": "1","DNT": "1","Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8","Accept-Language": "en-US,en;q=0.5","Accept-Encoding": "gzip, deflate"} 

main = requests.get(site, headers=header)
print(main.status_code)
src = main.content
soup = BeautifulSoup(src, 'lxml')

shoes = dict()
blocks = list(soup.find_all('div', {'class': 'dsg-flex flex-column dsg-react-product-card rs_product_card'}))
print(len(blocks))
counter = 1
for block in blocks:
	specs = block.find('div', {'class': 'dsg-flex flex-column rs_card_layout'})
	description = specs.find('a', {'class': 'rs_product_description d-block'})
	name = description.text
	print('Product Name: ' + name)
	shoe_link = homepage + description.attrs['href']
	print('Link: ' + shoe_link)
	price_tag = specs.find('div', {'class': 'rs_product_price'})
	try:
		orig_price = price_tag.find('p', {'class': 'was-price'}).text
	except:
		orig_price = price_tag.find('p', {'class': 'final-price'}).text
	try: 
		orig_price = float(orig_price.split('-')[0].split(':')[1].strip().replace('$', '').replace('*', ''))
	except:
		orig_price = float(orig_price.split('-')[0].split(':')[0].strip().replace('$', '').replace('*', ''))
	print('Original Price: ' + str(orig_price))
	try:
		current_price = price_tag.find('p', {'class': 'offer-price'}).text[1:]
		current_price = float(current_price.split('-')[0].strip())
	except:
		current_price = orig_price
	print('Current Price: ' + str(current_price))
	discount = round(orig_price - current_price, 3)
	print('Difference: ' + str(discount))
	shoes[counter] = [name, shoe_link, current_price, discount]
	counter += 1
	print('___________________________________________')

for key, val in shoes.items():
	shoes[key] = list(map(lambda x: str(x), shoes[key]))
print(len(shoes))


import psycopg2

try:
	con = psycopg2.connect(
			host = "localhost", 
			database = "busysquirrels",
			user = 'not-user',
			password = 'not-password',
			port = 5432)
	cur = con.cursor()
	
	#Commented Out Portion Is Experimentation 
	"""
	cur.execute('SELECT * FROM bsTable')
	rows = cur.fetchall()
	exist_names = [row[1] for row in rows]
	new_rows = list()
	updated_rows = list()
	counter = 1
	for value in shoes.values():
		if value[0] in exist_names:
			#Add to updated_rows
			updated_rows.append(tuple(shoes[counter]))
			
		else:
			#Add to new_rows
			new_rows.append(tuple(shoes[counter]))
		counter +=1
		"""
	#print('New Rows: ', new_rows)
	#print('Updated Rows: ', updated_rows)
	"""
	ps_query_add = "INSERT INTO BusySquirrelsTable (NAME, LINK, CURRENTPRICE, DISCOUNT) VALUES (%s, %s, %s, %s)"
	for content in new_rows:
		name_ = content[0]
		link_ = content[1]
		curr_price_ = content[2]
		discount_ = content[3]
		record = (name_, link_, curr_price_, discount_)
		cur.execute(ps_query_add, record)
	
	
	con.commit()
	count = cur.rowcount
	print(count, 'record inserted to shoes table')
	
except Exception as e:
	print(e)
	print('Failed to insert to shoes table')

finally:
	if con:
		cur.close()
		con.close()
		print('Postgres Closed')
		
"""
	ps_query = "INSERT INTO BusySquirrelsTable (NAME, LINK, CURRENTPRICE, DISCOUNT) VALUES (%s, %s, %s, %s)"
	for content in shoes.values():
		name_ = content[0]
		link_ = content[1]
		curr_price_ = content[2]
		discount_ = content[3]
		record = (name_, link_, curr_price_, discount_)
		print('here!')
		cur.execute(ps_query, record)
	con.commit()
	count = cur.rowcount
	print(count, 'record inserted to shoes table')
except Exception as e:
	print(e)
	print('Failed to insert to shoes table')

finally:
	if con:
		cur.close()
		con.close()
		print('Postgres Closed')

#Then Uses Cron Job to Update Every 5 minutes
