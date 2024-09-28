import os
from faster_whisper import WhisperModel
from mutagen.flac import FLAC
from mutagen.mp3 import MP3

def transcribe_audio_files(directory):
    # Load the Whisper model (adjust model size and device as needed)
    model = WhisperModel('large-v3', device='cuda', compute_type='float16')

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.mp3') or filename.endswith('.wav') or filename.endswith('.flac'):  # Adjust file types as needed
            if filename.endswith('.mp3'):
                Format = "mp3"
                audio = MP3(filename)
                duration = audio.info.length
            elif filename.endswith('.flac'):
                Format = "flac"
                audio = FLAC(filename)
                duration = audio.info.length
            elif filename.endswith('.wav'):
                Format = "wav"
                with wave.open(filename, 'r') as wav:
                    frames = wav.getnframes()
                    rate = wav.getframerate()
                    duration = frames / float(rate)

            file_path = os.path.join(directory, filename)
            print(f"Transcribing {filename}...")

            # Transcribe the audio file
            segments, info = model.transcribe(file_path, beam_size=5)

            # Print detected language and its probability
            print("Detected language %s with probability %f" % (info.language, info.language_probability))

            # Print each segment's start time, end time, and text
            filename_editted = filename.replace(f'.{Format}' , '')
            with open(directory+"\\"+filename_editted+'.vtt' ,'w', encoding = 'utf-8') as file :
                file.write("WEBVTT \n\n")
                for segment in segments:
                    print(segment.start/duration)
                    file.write("%.2fs --> %.2fs \n%s" % (segment.start, segment.end, segment.text) + '\n\n')
            print("Finished")
if __name__ == "__main__":
    # Specify your directory containing audio files
    audio_directory = str(input("Input the directory : "))
    audio_directory2 = audio_directory.replace("/" , "\"")
    transcribe_audio_files(audio_directory2)
