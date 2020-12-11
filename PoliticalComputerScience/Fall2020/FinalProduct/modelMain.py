import os
from PIL import Image


def generate_text_seq(model, tokenizer, text_seq_length, seed_text, numWords):
	predText = []
	for _ in range(numWords):
		encoded = tokenizer.texts_to_sequences([seed_text])[0]
		encoded = pad_sequences([encoded], maxlen=text_seq_length, truncating='pre')
		y_pred = model.predict_classes(encoded)
		predWord = ''
		for word, index in tokenizer.word_index.items():
			if index == y_pred:
				predWord = word
				break
		seed_text = seed_text + ' ' + predWord
		predText.append(predWord)
	return ' '.join(predText)

def getTextGen(results):
	resultSentences = []
	for judge in results:
		tgModel = tf.keras.models.load_model(judge + "TG.h5")
		seed = groups[random.randrange(200)]
		sentences = []
		#As of now, it's judgesSentencesOnly.csv
		with open(judge + 'Sentences.csv', newline='') as file:
			reader = csv.reader(file, delimiter=',')
			for row in reader:
				sentences.append(row)
		sents = [sent[0] for sent in sentences if len(sent[0]) > 2]
		sents = " ".join(sents)
		def clean_text(data):
			words = data.split()
			table = str.maketrans('', '', string.punctuation)
			words = [word.translate(table) for word in words]
			words = [word for word in words if word.isalpha() ]
			words = [word.lower() for word in words]
			return words

		cleanSents = clean_text(sents)
		predLength = 15
		groups = []
		for i in range(predLength, len(cleanSents)):
			part = cleanSents[i - predLength: i]
			grouping = ' '.join(part)
			groups.append(grouping)
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(groups)
		result = generate_text_seq(rnnModel, tokenizer, 14, seed, 15)
		resultSentences.append(result)
	return resultSentences

results = pd.read_csv('ThreeJudgesSA.csv')
results = [results.iloc[:, i].values[0] for i in range(3)]
textGenResults = getTextGen(results)
textGenResultsDict = dict()
textGenResultsDict[results[0]] = textGenResults[0]
textGenResultsDict[results[1]] = textGenResults[1]
textGenResultsDict[results[2]] = textGenResults[2]

print('0) ', results[0])
print('1) ', results[1])
print('2) ', results[2])

inputResults = []

for i in range(2):
	print('Choose the Number Corresponding to the Quotation that Best Fits You.')
	x = str(input())
	if x == '0':
		inputResults.append(results[0])
	elif x == '1':
		inputResults.append(results[1])
	else:
		inputResults.append(results[2])
	

finalTwo = inputResults

first = True
for last in finalTwo:
	for pic in os.listdir('./JudgePics'):
		if first:
			print('First Image is ' + last)
			first = False
		else:
			print('Second Image is ' + last)
		im = Image.open('./JudgePics/' + last + '.jpg')
		im.show()
print('Which Picture Do You Identify Most With?')
choice = input('Choose 1 or 2')
print('You Were Matched With' + finalTwo[int(choice)-1]+ '!')
