import moviepy.editor as mp
import os
# import speech_recognition
import whisper
from pydub import AudioSegment
from moviepy.video.io.VideoFileClip import VideoFileClip

AUDIO_BASE_PATH = "data_1Min/audio/"
VIDEO_BASE_PATH = "data_1Min/video/"
RESULTS_BASE_PATH = "data_1Min/results/"
AUDIO_FILE_NAME = "full_audio.wav"

SHOULD_COMPRESS = True
# Divide into 30secs, 1 min, 
SPLIT_LENGTH_SECONDS = 1 * 60

MILLISECOND_CONVERT = 1000

PARTITIONS_SUBDIRECTORY_NAME = "partitions"

# r = speech_recognition.Recognizer()

asr_model = whisper.load_model("base")


def split_video(input_file_name):
    input_file_path = VIDEO_BASE_PATH + input_file_name + ".mp4"

    # Load the video file
    video = VideoFileClip(input_file_path)
    
    # Get the duration of the video in seconds
    output_folder = VIDEO_BASE_PATH + PARTITIONS_SUBDIRECTORY_NAME + "/" + input_file_name
    duration = int(video.duration)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Split the video into smaller clips
    for start_time in range(0, duration, SPLIT_LENGTH_SECONDS):
        end_time = min(start_time + SPLIT_LENGTH_SECONDS, duration)
        clip = video.subclip(start_time, end_time)
        output_file = f"{output_folder}/{start_time}-{end_time}.mp4"
        clip.write_videofile(output_file, codec="libx264")
    
    video.close()

def extract_audio_from_video(video_path: str, audio_path: str) -> None:
    print("Extracting audio from video")
    video_clip = mp.VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)


def extract_speech(audio_file: str) -> str:
    print("Extracting text from audio for file " + audio_file)
    transcription = asr_model.transcribe(audio_file)
    if not transcription or 'text' not in transcription:
        return ""
    return transcription['text']
    

def split_audio(input_file_path: str, split_length: str, compress: bool) -> None:
    print(f'Splitting audio into {SPLIT_LENGTH_SECONDS} second intervals')
    split_length_ms = split_length * MILLISECOND_CONVERT
    input_file = input_file_path + AUDIO_FILE_NAME
    partitions_path = input_file_path + PARTITIONS_SUBDIRECTORY_NAME
    if not os.path.exists(partitions_path):
        os.makedirs(partitions_path)

    sound = AudioSegment.from_wav(input_file)
    total_length_ms = len(sound)


    num_splits = total_length_ms // split_length_ms
    remainder = total_length_ms % split_length_ms
    end_time = 0


    for i in range(num_splits):
        start_time = i * split_length_ms
        end_time = (i + 1) * split_length_ms
        split = sound[start_time:end_time]
        if compress:
            split = split.set_frame_rate(32000).set_channels(1).set_sample_width(1)
        split.export(f"{partitions_path}/{start_time//MILLISECOND_CONVERT}-{end_time//MILLISECOND_CONVERT}.wav", format="wav")

    if remainder != 0:
        last_split = sound[-remainder:]
        if compress:
            split = split.set_frame_rate(32000).set_channels(1).set_sample_width(1)
        last_split.export(f"{partitions_path}/{end_time//MILLISECOND_CONVERT}-{total_length_ms//MILLISECOND_CONVERT}.wav", format="wav")