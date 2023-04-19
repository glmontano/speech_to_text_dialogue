import whisper # pip install -U openai-whisper
import json

from pyannote.audio import Pipeline # pip install -qq https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
from re import findall
from pydub import AudioSegment # pip install pydub

SOUND_FILE = 'customer_service_sample.mp3'
SOUND_FILE_FORMAT = SOUND_FILE.split('.')[-1]

ACCESS_TOKEN = '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'
MODEL_KEY = 'small.en'

TEXT_LOG = 'text_output.txt'

SPEAKER_DICT = {'SPEAKER_01': 'Customer Service',
				'SPEAKER_02': 'Client'}


def diaritize():

	# Pyannote (Likely getting paramters from specified location)
	pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=ACCESS_TOKEN)

	# 4. apply pretrained pipeline
	diarization = pipeline(SOUND_FILE)

	# 5. Write to text file.
	with open(TEXT_LOG, "w") as text_file:
	    text_file.write(str(diarization))

    # # 5. print the result
	# for turn, _, speaker in diarization.itertracks(yield_label=True):
	#     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


def millisec(timeStr):

	spl = timeStr.split(":")
	s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
	return s


def process_times():

	dzs = open(TEXT_LOG).read().splitlines()

	groups = []
	g = []
	lastend = 0

	for d in dzs:

		#same speaker
		if g and (g[0].split()[-1] != d.split()[-1]):      
			groups.append(g)
			g = []

		g.append(d)

		end = findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
		end = millisec(end)

		#segment engulfed by a previous segment
		if (lastend > end):       
			groups.append(g)
			g = [] 
		else:
			lastend = end

	if g:
		groups.append(g)

	# print(*groups, sep='\n')
	# print(groups)

	return groups

def split_audio(time_groups):

	# Split audio
	audio = AudioSegment.from_mp3(SOUND_FILE)
	gidx = -1
	for g in time_groups:
		start = findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
		end = findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
		start = millisec(start) #- spacermilli
		end = millisec(end)  #- spacermilli
		gidx += 1
		audio[start:end].export(f'{str(gidx)}.{SOUND_FILE_FORMAT}', format=SOUND_FILE_FORMAT)
		print(f"group {gidx}: {start}--{end}")

def whisper_transcribe(time_groups):

	model = whisper.load_model(MODEL_KEY)
	gidx = -1
	for i in range(len(time_groups)):
		gidx += 1
		audiof = f'{str(gidx)}.{SOUND_FILE_FORMAT}'
		result = model.transcribe(audio=audiof, fp16=False, language='en', word_timestamps=True)#, initial_prompt=result.get('text', ""))
		with open(str(i)+'.json', "w") as outfile:
			json.dump(result, outfile, indent=4)  


# diaritize()
the_times = process_times()
# split_audio(the_times)
whisper_transcribe(the_times)
