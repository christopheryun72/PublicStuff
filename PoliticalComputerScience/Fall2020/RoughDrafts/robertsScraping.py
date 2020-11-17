#Christopher Yun
#@christopheryun72@berkeley.edu

import requests
from bs4 import BeautifulSoup
import pandas as pd

result = requests.get('https://www.law.cornell.edu/supct/justices/roberts.dec.html')
print(result.status_code)
src = result.content
soup = BeautifulSoup(src, 'lxml')

cases = []
caseList = soup.find_all('ul')
caseList = list(caseList[8])[2:]

for li_tag in caseList:
	a_tag = li_tag.find("a")
	cases.append(a_tag.attrs['href'])

def find_between( s, first, last ):
	try:
		start = s.index( first ) + len( first )
		end = s.index( last, start )
		return s[start:end]
	except ValueError:
		return ""

finalOpinion = []

for url in cases[::10]:
	try:
		#print('https://www.law.cornell.edu/supct/justices/thomas.dec.html' + url)
		print(url)
		case = requests.get('https://www.law.cornell.edu' + url)
		caseSRC = case.content
		soupCase = BeautifulSoup(caseSRC, 'lxml')
		#sent = soupCase.find_all('p')
		sent = soupCase.text
		sent = " ".join(sent.split())
		opinion = sent[sent.index("Justice ROBERTS delivered the opinion of the Court.") + len("Justice ROBERTS delivered the opinion of the Court."): ]
		opinionList = opinion.split(".")
		finalOpinion.append(opinionList)
		"""
		sent = soupCase.find_all(text=True)
		print('p' in set([t.parent.name for t in sent]))
		print([p.text for p in soupCase.findAll('p')])	
		"""
	except:
		continue

final = []
for case in finalOpinion:
	temp = []
	for val in case:
		if len(val) > 2:
			temp.append(val)
	final.append(temp)

print(final)

finalFinal = []
for lst in final:
	temp = lst[:int(len(lst)/2)]
	finalFinal.append(temp)
	
done = []
for lst in finalFinal:
	temp = []
	for val in lst:
		numSpaces = val.count(' ')
		if numSpaces > 3:
			temp.append(val)
	done.append(temp)

finalFinal = done

for lst in finalFinal:
	print(len(lst))

excelReady = [j for sub in finalFinal for j in sub]

roberts = ["Roberts"] * len(excelReady)

df = pd.DataFrame()
df['Sentence'] = excelReady[:]
df['Judge'] = roberts[:]

df.to_excel('chrisJudges2.xlsx', index = False)