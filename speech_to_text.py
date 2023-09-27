import argparse
import json
import requests
import os
import vlc
import time

"""
Working version of text to speech using Eleven Labs API.
Just run the script with the text to be converted as a command line argument.
"""


# text as command line argument with instant output 

def text_to_speech(text):
    url = 'https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB/stream'
    headers = {
        'accept': 'audio/mpeg',
        'xi-api-key': 'fe948e1825744b2e25f9ac0a161d4698',
        'Content-Type': 'application/json'
    }
    data = {
        'text': text,
        'voice_settings': {
            'stability': 0.4,
            'similarity_boost': 0.75
        }
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

    if response.status_code == 200 and response.headers.get('content-type') == 'audio/mpeg':
        temp_file_name = 'temp_audio.mp3'
        
        # Write the audio data to the temporary file
        with open(temp_file_name, 'wb') as temp_file:
            for chunk in response.iter_content(chunk_size=1024):
                temp_file.write(chunk)

        instance = vlc.Instance('--no-plugins-cache')
        player = instance.media_player_new()

        # Use the temporary file as the media source
        media = instance.media_new(temp_file_name)
        player.set_media(media)
        player.play()

        while player.get_state() != vlc.State.Ended:
            pass
        time.sleep(0.1)  # Add a short delay before removing the file

        # Remove the temporary file after playing the audio
        os.remove(temp_file_name)

        return 'success'
    else:
        return 'fail'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text to Speech using Eleven Labs API')
    parser.add_argument('--text', type=str, required=True, help='Text to be converted to speech')
    args = parser.parse_args()

    result = text_to_speech(args.text)
    print(result)